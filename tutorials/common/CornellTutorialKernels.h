#include <hiprt/hiprt_device.h>
#include <hiprt/hiprt_math.h>
#include <hiprt/hiprt_vec.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 64
#endif

#ifndef SHARED_STACK_SIZE
#define SHARED_STACK_SIZE 16
#endif

HIPRT_DEVICE HIPRT_INLINE uint32_t lcg( uint32_t& seed )
{
	constexpr uint32_t A = 1103515245u;
	constexpr uint32_t C = 12345u;
	constexpr uint32_t M = 0x00FFFFFFu;
	seed = A * seed + C;
	return seed & M;
}

HIPRT_DEVICE HIPRT_INLINE float randf( uint32_t& seed )
{
	return static_cast<float>( lcg( seed ) ) / static_cast<float>( 0x01000000 );
}

HIPRT_DEVICE HIPRT_INLINE float3 sampleHemisphereCosine( float3 n, uint32_t& seed )
{
	const float phi			 = 2.0f * hiprt::Pi * randf( seed );
	const float sinThetaSqr = randf( seed );
	const float sinTheta	 = sqrtf( sinThetaSqr );

	const float3 axis = fabsf( n.x ) > 0.001f ? float3{ 0.0f, 1.0f, 0.0f } : float3{ 1.0f, 0.0f, 0.0f };
	float3		  t	   = hiprt::cross( axis, n );
	t			   = hiprt::normalize( t );
	const float3 s = hiprt::cross( n, t );

	return hiprt::normalize( s * cosf( phi ) * sinTheta + t * sinf( phi ) * sinTheta + n * sqrtf( 1.0f - sinThetaSqr ) );
}

extern "C" __global__ void CornellBoxKernel( hiprtGeometry geom, uint8_t* pixels, hiprtInt2 res )
{
	const uint32_t x	 = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y	 = blockIdx.y * blockDim.y + threadIdx.y;
	if ( x >= static_cast<uint32_t>( res.x ) || y >= static_cast<uint32_t>( res.y ) ) return;

	const uint32_t index = x + y * res.x;

	hiprtRay ray;
	ray.origin = { 278.0f, 273.0f, -900.0f };
	const float2 d = { 2.0f * x / static_cast<float>( res.x ) - 1.0f, 2.0f * y / static_cast<float>( res.y ) - 1.0f };
	const float3 uvw = { -387.817566f, -387.817566f, 1230.0f };
	ray.direction = { uvw.x * d.x, uvw.y * d.y, uvw.z };
	ray.direction =
		ray.direction /
		sqrtf( ray.direction.x * ray.direction.x + ray.direction.y * ray.direction.y + ray.direction.z * ray.direction.z );

	hiprtGeomTraversalClosest tr( geom, ray );
	hiprtHit				  hit = tr.getNextHit();

	int3 color = { 0, 0, 0 };
	if ( hit.hasHit() )
	{
		const float3 n = hiprt::normalize( hit.normal );
		color.x		   = static_cast<int>( ( ( n.x + 1.0f ) * 0.5f ) * 255.0f );
		color.y		   = static_cast<int>( ( ( n.y + 1.0f ) * 0.5f ) * 255.0f );
		color.z		   = static_cast<int>( ( ( n.z + 1.0f ) * 0.5f ) * 255.0f );
	}

	pixels[index * 4 + 0] = min( 255, color.x );
	pixels[index * 4 + 1] = min( 255, color.y );
	pixels[index * 4 + 2] = min( 255, color.z );
	pixels[index * 4 + 3] = 255;
}

extern "C" __global__ void
SharedStackKernel( hiprtGeometry geom, uint8_t* pixels, hiprtInt2 res, hiprtGlobalStackBuffer globalStackBuffer )
{
	const uint32_t x	 = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y	 = blockIdx.y * blockDim.y + threadIdx.y;
	if ( x >= static_cast<uint32_t>( res.x ) || y >= static_cast<uint32_t>( res.y ) ) return;

	const uint32_t index = x + y * res.x;

	hiprtRay ray;
	ray.origin = { 278.0f, 273.0f, -900.0f };
	const float2 d = { 2.0f * x / static_cast<float>( res.x ) - 1.0f, 2.0f * y / static_cast<float>( res.y ) - 1.0f };
	const float3 uvw = { -387.817566f, -387.817566f, 1230.0f };
	ray.direction = { uvw.x * d.x, uvw.y * d.y, uvw.z };
	ray.direction =
		ray.direction /
		sqrtf( ray.direction.x * ray.direction.x + ray.direction.y * ray.direction.y + ray.direction.z * ray.direction.z );

	__shared__ int		   sharedStackCache[SHARED_STACK_SIZE * BLOCK_SIZE];
	hiprtSharedStackBuffer sharedStackBuffer{ SHARED_STACK_SIZE, sharedStackCache };

	hiprtGlobalStack									   stack( globalStackBuffer, sharedStackBuffer );
	hiprtGeomTraversalClosestCustomStack<hiprtGlobalStack> tr( geom, ray, stack );
	hiprtHit											   hit = tr.getNextHit();

	int3 color = { 0, 0, 0 };
	if ( hit.hasHit() )
	{
		const float3 n = hiprt::normalize( hit.normal );
		color.x		   = static_cast<int>( ( ( n.x + 1.0f ) * 0.5f ) * 255.0f );
		color.y		   = static_cast<int>( ( ( n.y + 1.0f ) * 0.5f ) * 255.0f );
		color.z		   = static_cast<int>( ( ( n.z + 1.0f ) * 0.5f ) * 255.0f );
	}

	pixels[index * 4 + 0] = min( 255, color.x );
	pixels[index * 4 + 1] = min( 255, color.y );
	pixels[index * 4 + 2] = min( 255, color.z );
	pixels[index * 4 + 3] = 255;
}

extern "C" __global__ void AmbientOcclusionKernel( hiprtGeometry geom, uint8_t* pixels, hiprtInt2 res, float aoRadius )
{
	const uint32_t x	 = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y	 = blockIdx.y * blockDim.y + threadIdx.y;
	if ( x >= static_cast<uint32_t>( res.x ) || y >= static_cast<uint32_t>( res.y ) ) return;

	const uint32_t index = x + y * res.x;
	constexpr uint32_t Spp = 8u;
	constexpr uint32_t AoSamples = 8u;
	float ao = 0.0f;

	for ( uint32_t p = 0; p < Spp; ++p )
	{
		uint32_t seed = x + y * res.x + p * 9781u;
		hiprtRay ray;
		ray.origin = { 278.0f, 273.0f, -900.0f };
		const float2 d = {
			2.0f * ( x + randf( seed ) ) / static_cast<float>( res.x ) - 1.0f,
			2.0f * ( y + randf( seed ) ) / static_cast<float>( res.y ) - 1.0f };
		const float3 uvw = { -387.817566f, -387.817566f, 1230.0f };
		ray.direction = { uvw.x * d.x, uvw.y * d.y, uvw.z };
		ray.direction =
			ray.direction /
			sqrtf( ray.direction.x * ray.direction.x + ray.direction.y * ray.direction.y + ray.direction.z * ray.direction.z );

		hiprtGeomTraversalClosest tr( geom, ray );
		const hiprtHit hit = tr.getNextHit();
		if ( !hit.hasHit() ) continue;

		const float3 surfacePt = ray.origin + hit.t * ( 1.0f - 1.0e-2f ) * ray.direction;
		float3 Ng = hit.normal;
		if ( hiprt::dot( ray.direction, Ng ) > 0.0f ) Ng = -Ng;
		Ng = hiprt::normalize( Ng );

		for ( uint32_t i = 0; i < AoSamples; ++i )
		{
			hiprtRay aoRay;
			aoRay.origin = surfacePt;
			aoRay.direction = sampleHemisphereCosine( Ng, seed );
			aoRay.maxT = aoRadius;

			hiprtGeomTraversalAnyHit aoTr( geom, aoRay );
			const hiprtHit aoHit = aoTr.getNextHit();
			ao += !aoHit.hasHit() ? 1.0f : 0.0f;
		}
	}

	ao /= ( Spp * AoSamples );
	const uint8_t value = static_cast<uint8_t>( ao * 255.0f );
	pixels[index * 4 + 0] = value;
	pixels[index * 4 + 1] = value;
	pixels[index * 4 + 2] = value;
	pixels[index * 4 + 3] = 255;
}
