//
// Copyright (c) 2021-2023 Advanced Micro Devices, Inc. All rights reserved.
//

#include <filesystem>
#include <tutorials/common/Aabb.h>
#include <tutorials/common/FluidSimulation.h>
#include <tutorials/common/TutorialBase.h>

class Tutorial : public TutorialBase
{
  public:
	static constexpr std::string_view OutputDir		  = "fluid_simulation_output";
	static constexpr bool			  ExportAllFrames = false;
	static constexpr uint32_t		  FrameCount	  = 200u;
	static constexpr uint32_t		  FrameToExport	  = 165u;

	static constexpr float PoolVolumeDim			  = 1.0f;
	static constexpr float PoolSpaceDivision		  = 50.0f;
	static constexpr float InitParticleVolumeDim	  = 0.6f;
	static constexpr float InitParticleVolumeCenter[] = {
		-0.45f * ( PoolVolumeDim - InitParticleVolumeDim ),
		PoolVolumeDim - InitParticleVolumeDim * 0.5f,
		0.45f * ( PoolVolumeDim - InitParticleVolumeDim ) };
	static constexpr float ParticleRestDensity	= 1000.0f;
	static constexpr float ParticleSmoothRadius = PoolVolumeDim / PoolSpaceDivision;

	void run()
	{
		Simulation sim{};
		sim.m_smoothRadius		= ParticleSmoothRadius;
		sim.m_pressureStiffness = 200.0f;
		sim.m_restDensity		= ParticleRestDensity;
		sim.m_wallStiffness		= 3000.0f;
		sim.m_particleCount		= 131072;
		sim.m_planes[0]			= { 0.0f, 1.0f, 0.0f, 0.0f };
		sim.m_planes[1]			= { 0.0f, -1.0f, 0.0f, PoolVolumeDim };
		sim.m_planes[2]			= { 1.0f, 0.0f, 0.0f, 0.5f * PoolVolumeDim };
		sim.m_planes[3]			= { -1.0f, 0.0f, 0.0f, 0.5f * PoolVolumeDim };
		sim.m_planes[4]			= { 0.0f, 0.0f, 1.0f, 0.5f * PoolVolumeDim };
		sim.m_planes[5]			= { 0.0f, 0.0f, -1.0f, 0.5f * PoolVolumeDim };

		const float initVolume = InitParticleVolumeDim * InitParticleVolumeDim * InitParticleVolumeDim;
		const float mass = sim.m_restDensity * initVolume / sim.m_particleCount;
		const float viscosity = 0.4f;
		sim.m_densityCoef = mass * 315.0f / ( 64.0f * hiprt::Pi * pow( sim.m_smoothRadius, 9.0f ) );
		sim.m_pressureGradCoef = mass * -45.0f / ( hiprt::Pi * pow( sim.m_smoothRadius, 6.0f ) );
		sim.m_viscosityLaplaceCoef = mass * viscosity * 45.0f / ( hiprt::Pi * pow( sim.m_smoothRadius, 6.0f ) );

		std::vector<Particle> particles( sim.m_particleCount );
		const auto smoothRadius = ParticleSmoothRadius;
		const auto dimSize = static_cast<uint32_t>( ceil( cbrtf( static_cast<float>( sim.m_particleCount ) ) ) );
		const auto slcSize = dimSize * dimSize;
		for ( uint32_t i = 0; i < sim.m_particleCount; ++i )
		{
			const auto n = i % slcSize;
			auto x = ( n % dimSize ) / static_cast<float>( dimSize );
			auto y = ( n / dimSize ) / static_cast<float>( dimSize );
			auto z = ( i / slcSize ) / static_cast<float>( dimSize );
			x = InitParticleVolumeDim * ( x - 0.5f ) + InitParticleVolumeCenter[0];
			y = InitParticleVolumeDim * ( y - 0.5f ) + InitParticleVolumeCenter[1];
			z = InitParticleVolumeDim * ( z - 0.5f ) + InitParticleVolumeCenter[2];
			particles[i].Pos = { x, y, z };
			particles[i].Velocity = { 0.0f, 0.0f, 0.0f };
		}

		hiprtContext ctxt;
		CHECK_HIPRT( hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, ctxt ) );

		hiprtAABBListPrimitive list{};
		list.aabbCount = sim.m_particleCount;
		list.aabbStride = sizeof( Aabb );
		std::vector<Aabb> aabbs( sim.m_particleCount );
		for ( uint32_t i = 0; i < sim.m_particleCount; ++i )
		{
			const hiprtFloat3& c = particles[i].Pos;
			aabbs[i].m_min = { c.x - smoothRadius, c.y - smoothRadius, c.z - smoothRadius };
			aabbs[i].m_max = { c.x + smoothRadius, c.y + smoothRadius, c.z + smoothRadius };
		}
		CHECK_ORO( cudaMalloc( reinterpret_cast<void**>( &list.aabbs ), sim.m_particleCount * sizeof( Aabb ) ) );
		CHECK_ORO( cuMemcpyHtoD( reinterpret_cast<CUdeviceptr>( list.aabbs ), aabbs.data(), sim.m_particleCount * sizeof( Aabb ) ) );

		hiprtGeometryBuildInput geomInput{};
		geomInput.type = hiprtPrimitiveTypeAABBList;
		geomInput.primitive.aabbList = list;
		geomInput.geomType = 0;

		size_t geomTempSize = 0;
		hiprtDevicePtr geomTemp = nullptr;
		hiprtBuildOptions options{};
		options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
		CHECK_HIPRT( hiprtGetGeometryBuildTemporaryBufferSize( ctxt, geomInput, options, geomTempSize ) );
		CHECK_ORO( cudaMalloc( reinterpret_cast<void**>( &geomTemp ), geomTempSize ) );

		hiprtGeometry geom = nullptr;
		CHECK_HIPRT( hiprtCreateGeometry( ctxt, geomInput, options, geom ) );

		hiprtFuncNameSet funcNameSet{};
		funcNameSet.intersectFuncName = "intersectParticleImpactSphere";
		std::vector<hiprtFuncNameSet> funcNameSets = { funcNameSet };

		Particle* dParticles = nullptr;
		CHECK_ORO( cudaMalloc( reinterpret_cast<void**>( &dParticles ), sim.m_particleCount * sizeof( Particle ) ) );
		CHECK_ORO( cuMemcpyHtoD( reinterpret_cast<CUdeviceptr>( dParticles ), particles.data(), sim.m_particleCount * sizeof( Particle ) ) );
		hiprtFuncDataSet funcDataSet{};
		funcDataSet.intersectFuncData = dParticles;

		hiprtFuncTable funcTable;
		CHECK_HIPRT( hiprtCreateFuncTable( ctxt, 1, 1, funcTable ) );
		CHECK_HIPRT( hiprtSetFuncTable( ctxt, funcTable, 0, 0, funcDataSet ) );

		Simulation* pSim = nullptr;
		CHECK_ORO( cudaMalloc( reinterpret_cast<void**>( &pSim ), sizeof( Simulation ) ) );
		CHECK_ORO( cuMemcpyHtoD( reinterpret_cast<CUdeviceptr>( pSim ), &sim, sizeof( Simulation ) ) );

		float* densities = nullptr;
		hiprtFloat3* accelerations = nullptr;
		PerFrame* pPerFrame = nullptr;
		float4x4* pViewProj = nullptr;
		uint8_t* pixels = nullptr;

		CHECK_ORO( cudaMalloc( reinterpret_cast<void**>( &densities ), sim.m_particleCount * sizeof( float ) ) );
		CHECK_ORO( cudaMalloc( reinterpret_cast<void**>( &accelerations ), sim.m_particleCount * sizeof( hiprtFloat3 ) ) );
		CHECK_ORO( cudaMalloc( reinterpret_cast<void**>( &pPerFrame ), sizeof( PerFrame ) ) );
		CHECK_ORO( cudaMalloc( reinterpret_cast<void**>( &pViewProj ), sizeof( float4x4 ) ) );
		CHECK_ORO( cudaMalloc( reinterpret_cast<void**>( &pixels ), m_res.x * m_res.y * 4 ) ) ;

		PerFrame perFrame{};
		perFrame.m_timeStep = 1.0f / 320.0f;
		perFrame.m_gravity = { 0.0f, -9.8f, 0.0f };
		CHECK_ORO( cuMemcpyHtoD( reinterpret_cast<CUdeviceptr>( pPerFrame ), &perFrame, sizeof( PerFrame ) ) );

		const float aspect = m_res.x / static_cast<float>( m_res.y );
		const float4x4 proj = Perspective( hiprt::Pi / 4.0f, aspect, 1.0f, 40.0f );
		const hiprtFloat3 focusPt = { 0.0f, 0.5f, 0.0f };
		const hiprtFloat3 eyePt   = focusPt - hiprtFloat3{ -0.5f, -0.5f, 2.0f };
		const float4x4 view = LookAt( eyePt, focusPt, { 0.0f, 1.0f, 0.0f } );
		const float4x4 viewProj = proj * view;
		CHECK_ORO( cuMemcpyHtoD( reinterpret_cast<CUdeviceptr>( pViewProj ), &viewProj, sizeof( float4x4 ) ) );

		CUfunction densityFunc = nullptr;
		CUfunction forceFunc = nullptr;
		CUfunction intFunc = nullptr;
		CUfunction visFunc = nullptr;
		auto kernelPath = std::filesystem::path( HIPRTSDK_ROOT_DIR ) / "tutorials/common/FluidTutorialKernels.h";
		buildTraceKernel( ctxt, kernelPath, "DensityKernel", densityFunc, nullptr, &funcNameSets, 1, 1 );
		buildTraceKernel( ctxt, kernelPath, "ForceKernel", forceFunc, nullptr, &funcNameSets, 1, 1 );
		buildTraceKernel( ctxt, kernelPath, "IntegrationKernel", intFunc );
		buildTraceKernel( ctxt, kernelPath, "VisualizationKernel", visFunc );

		const uint32_t block = 64u;
		const uint32_t grid  = ( sim.m_particleCount + block - 1 ) / block;

		if constexpr ( ExportAllFrames ) std::filesystem::create_directory( OutputDir );

		for ( uint32_t i = 0; i < FrameCount; ++i )
		{
			CHECK_HIPRT( hiprtBuildGeometry( ctxt, hiprtBuildOperationBuild, geomInput, options, geomTemp, 0, geom ) );

			void* dArgs[] = { &geom, &densities, &dParticles, &pSim, &funcTable };
			launchKernel( densityFunc, grid, 1, block, 1, dArgs );

			void* fArgs[] = { &geom, &accelerations, &dParticles, &densities, &pSim, &funcTable };
			launchKernel( forceFunc, grid, 1, block, 1, fArgs );

			void* iArgs[] = { &dParticles, &list.aabbs, &accelerations, &pSim, &pPerFrame };
			launchKernel( intFunc, grid, 1, block, 1, iArgs );

			CHECK_ORO( cuMemsetD8( reinterpret_cast<CUdeviceptr>( pixels ), 0, m_res.x * m_res.y * 4 ) );
			void* vArgs[] = { &dParticles, &densities, &pixels, &m_res, &pViewProj };
			launchKernel( visFunc, grid, 1, block, 1, vArgs );

			if constexpr ( ExportAllFrames )
			{
				const auto imageName = std::string( OutputDir ) + "/" + std::to_string( i ) + ".png";
				writeImage( imageName, m_res.x, m_res.y, pixels );
			}
			else
			{
				std::cout << "Fluid simulation: frame " << i << " done." << std::endl;
				if ( i == FrameToExport ) writeImage( "16_fluid_simulation.png", m_res.x, m_res.y, pixels );
			}
		}

		CHECK_ORO( cudaFree( dParticles ) );
		CHECK_ORO( cudaFree( list.aabbs ) );
		CHECK_ORO( cudaFree( geomTemp ) );
		CHECK_ORO( cudaFree( pSim ) );
		CHECK_ORO( cudaFree( densities ) );
		CHECK_ORO( cudaFree( accelerations ) );
		CHECK_ORO( cudaFree( pPerFrame ) );
		CHECK_ORO( cudaFree( pViewProj ) );
		CHECK_ORO( cudaFree( pixels ) );
		CHECK_HIPRT( hiprtDestroyFuncTable( ctxt, funcTable ) );
		CHECK_HIPRT( hiprtDestroyGeometry( ctxt, geom ) );
		CHECK_HIPRT( hiprtDestroyContext( ctxt ) );
	}
};

int main( int argc, char** argv )
{
	Tutorial tutorial;
	tutorial.init( 0 );
	tutorial.run();
	return 0;
}
