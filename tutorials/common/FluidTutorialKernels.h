#include <common/Aabb.h>
#include <common/Common.h>
#include <common/FluidSimulation.h>

#include <hiprt/hiprt_device.h>
#include <hiprt/hiprt_math.h>
#include <hiprt/hiprt_vec.h>

HIPRT_DEVICE bool intersectParticleImpactSphere( const hiprtRay& ray, const void* data, void* payload, hiprtHit& hit )
{
	const float3 from = ray.origin;
	const Particle particle = reinterpret_cast<const Particle*>( data )[hit.primID];
	const Simulation* sim = reinterpret_cast<const Simulation*>( payload );
	const float3 center = particle.Pos;
	const float radius = sim->m_smoothRadius;

	const float3 d = center - from;
	const float r2 = hiprt::dot( d, d );
	if ( r2 >= radius * radius ) return false;

	hit.t = r2;
	hit.normal = d;
	return true;
}

HIPRT_DEVICE bool intersectFunc(
	uint32_t geomType,
	uint32_t rayType,
	const hiprtFuncTableHeader& tableHeader,
	const hiprtRay& ray,
	void* payload,
	hiprtHit& hit )
{
	(void)rayType;
	if ( geomType != 0 ) return false;
	const void* data = tableHeader.funcDataSets[0].intersectFuncData;
	return intersectParticleImpactSphere( ray, data, payload, hit );
}

HIPRT_DEVICE bool filterFunc(
	uint32_t geomType,
	uint32_t rayType,
	const hiprtFuncTableHeader& tableHeader,
	const hiprtRay& ray,
	void* payload,
	const hiprtHit& hit )
{
	(void)geomType;
	(void)rayType;
	(void)tableHeader;
	(void)ray;
	(void)payload;
	(void)hit;
	return false;
}

__device__ float calculateDensity( float r2, float h, float densityCoef )
{
	const float d2 = h * h - r2;
	return densityCoef * d2 * d2 * d2;
}

__device__ float calculatePressure( float rho, float rho0, float pressureStiffness )
{
	const float rhoRatio = rho / rho0;
	return pressureStiffness * max( rhoRatio * rhoRatio * rhoRatio - 1.0f, 0.0f );
}

__device__ float3
calculateGradPressure( float r, float d, float pressure, float pressure_j, float rho_j, float3 disp, float pressureGradCoef )
{
	const float avgPressure = 0.5f * ( pressure + pressure_j );
	return pressureGradCoef * avgPressure * d * d * disp / ( rho_j * r );
}

__device__ float3
calculateVelocityLaplace( float d, float3 velocity, float3 velocity_j, float rho_j, float viscosityLaplaceCoef )
{
	const float3 velDisp = velocity_j - velocity;
	return viscosityLaplaceCoef * d * velDisp / rho_j;
}

extern "C" __global__ void
DensityKernel( hiprtGeometry geom, float* densities, const Particle* particles, Simulation* sim, hiprtFuncTable table )
{
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if ( index >= sim->m_particleCount ) return;

	const Particle particle = particles[index];
	hiprtRay ray;
	ray.origin = particle.Pos;
	ray.direction = { 0.0f, 0.0f, 1.0f };
	ray.minT = 0.0f;
	ray.maxT = 0.0f;

	hiprtGeomCustomTraversalAnyHit tr( geom, ray, hiprtTraversalHintDefault, sim, table );

	float rho = 0.0f;
	while ( tr.getCurrentState() != hiprtTraversalStateFinished )
	{
		const hiprtHit hit = tr.getNextHit();
		if ( !hit.hasHit() ) continue;
		rho += calculateDensity( hit.t, sim->m_smoothRadius, sim->m_densityCoef );
	}

	densities[index] = rho;
}

extern "C" __global__ void ForceKernel(
	hiprtGeometry geom,
	float3* accelerations,
	const Particle* particles,
	const float* densities,
	Simulation* sim,
	hiprtFuncTable table )
{
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if ( index >= sim->m_particleCount ) return;

	const Particle particle = particles[index];
	const float rho = densities[index];
	hiprtRay ray;
	ray.origin = particle.Pos;
	ray.direction = { 0.0f, 0.0f, 1.0f };
	ray.minT = 0.0f;
	ray.maxT = 0.0f;

	const float pressure = calculatePressure( rho, sim->m_restDensity, sim->m_pressureStiffness );
	hiprtGeomCustomTraversalAnyHit tr( geom, ray, hiprtTraversalHintDefault, sim, table );

	float3 force = hiprt::make_float3( 0.0f );
	while ( tr.getCurrentState() != hiprtTraversalStateFinished )
	{
		const hiprtHit hit = tr.getNextHit();
		if ( !hit.hasHit() || hit.primID == index ) continue;

		const Particle hitParticle = particles[hit.primID];
		const float hitRho = densities[hit.primID];
		const float3 disp = hit.normal;
		const float r = sqrtf( hit.t );
		const float d = sim->m_smoothRadius - r;
		const float hitPressure = calculatePressure( hitRho, sim->m_restDensity, sim->m_pressureStiffness );

		force += calculateGradPressure( r, d, pressure, hitPressure, hitRho, disp, sim->m_pressureGradCoef );
		force += calculateVelocityLaplace( d, particle.Velocity, hitParticle.Velocity, hitRho, sim->m_viscosityLaplaceCoef );
	}

	accelerations[index] = rho > 0.0f ? force / rho : hiprt::make_float3( 0.0f );
}

extern "C" __global__ void
IntegrationKernel( Particle* particles, Aabb* particleAabbs, const float3* accelerations, const Simulation* sim, const PerFrame* perFrame )
{
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if ( index >= sim->m_particleCount ) return;

	Particle particle = particles[index];
	float3 acceleration = accelerations[index];

	for ( uint32_t i = 0; i < 6; ++i )
	{
		const float d = dot4( hiprt::make_float4( particle.Pos, 1.0f ), sim->m_planes[i] );
		acceleration += min( d, 0.0f ) * -sim->m_wallStiffness * hiprt::make_float3( sim->m_planes[i] );
	}

	acceleration += perFrame->m_gravity;
	particle.Velocity += perFrame->m_timeStep * acceleration;
	particle.Pos += perFrame->m_timeStep * particle.Velocity;

	Aabb aabb;
	aabb.m_min = particle.Pos - sim->m_smoothRadius;
	aabb.m_max = particle.Pos + sim->m_smoothRadius;

	particles[index] = particle;
	particleAabbs[index] = aabb;
}

extern "C" __global__ void
VisualizationKernel( const Particle* particles, const float* densities, uint8_t* pixels, int2 res, const float4x4* viewProj )
{
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;

	const Particle particle = particles[index];
	const float rho = densities[index];

	float4 pos = ( *viewProj ) * hiprt::make_float4( particle.Pos, 1.0f );
	pos.x /= pos.w;
	pos.y /= pos.w;
	pos.z /= pos.w;

	const int x = ( pos.x * 0.5f + 0.5f ) * res.x;
	const int y = ( 0.5f - pos.y * 0.5f ) * res.y;
	if ( x < 0 || x >= res.x || y < 0 || y >= res.y ) return;

	const float visRho = rho / 4000.0f;
	const int pixelIndex = x + y * res.x;
	pixels[pixelIndex * 4 + 0] = visRho * 255.0f;
	pixels[pixelIndex * 4 + 1] = 0;
	pixels[pixelIndex * 4 + 2] = ( 1.0f - visRho ) * 255.0f;
	pixels[pixelIndex * 4 + 3] = 255;
}
