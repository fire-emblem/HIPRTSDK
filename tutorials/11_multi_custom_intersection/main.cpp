//
// Copyright (c) 2021-2023 Advanced Micro Devices, Inc. All rights reserved.
//

#include <tutorials/common/TutorialBase.h>

class Tutorial : public TutorialBase
{
  public:
	void run()
	{
		hiprtContext ctxt;
		CHECK_HIPRT( hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, ctxt ) );

		hiprtGeometry geoms[2]{};

		hiprtAABBListPrimitive listSpheres{};
		{
			const hiprtFloat4 sphere = { -0.15f, 0.0f, 0.0f, 0.15f };
			const hiprtFloat4 aabb[] = {
				{ sphere.x - sphere.w, sphere.y - sphere.w, sphere.z - sphere.w, 0.0f },
				{ sphere.x + sphere.w, sphere.y + sphere.w, sphere.z + sphere.w, 0.0f },
			};
			hiprtFloat4* dAabbs = nullptr;
			CHECK_ORO( cudaMalloc( reinterpret_cast<void**>( &dAabbs ), sizeof( aabb ) ) );
			listSpheres.aabbs = dAabbs;
			listSpheres.aabbCount = 1;
			listSpheres.aabbStride = 2 * sizeof( hiprtFloat4 );
			CHECK_ORO( cuMemcpyHtoD( reinterpret_cast<CUdeviceptr>( dAabbs ), aabb, sizeof( aabb ) ) );

			hiprtGeometryBuildInput geomInput{};
			geomInput.type = hiprtPrimitiveTypeAABBList;
			geomInput.primitive.aabbList = listSpheres;
			geomInput.geomType = 0;

			hiprtBuildOptions options{};
			options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
			size_t geomTempSize = 0;
			hiprtDevicePtr geomTemp = nullptr;
			CHECK_HIPRT( hiprtGetGeometryBuildTemporaryBufferSize( ctxt, geomInput, options, geomTempSize ) );
			CHECK_ORO( cudaMalloc( reinterpret_cast<void**>( &geomTemp ), geomTempSize ) );
			CHECK_HIPRT( hiprtCreateGeometry( ctxt, geomInput, options, geoms[0] ) );
			CHECK_HIPRT( hiprtBuildGeometry( ctxt, hiprtBuildOperationBuild, geomInput, options, geomTemp, 0, geoms[0] ) );
			CHECK_ORO( cudaFree( geomTemp ) );
		}

		hiprtAABBListPrimitive listCircles{};
		{
			const hiprtFloat4 circle = { 0.15f, 0.0f, 0.0f, 0.15f };
			const hiprtFloat4 aabb[] = {
				{ circle.x - circle.w, circle.y - circle.w, circle.z, 0.0f },
				{ circle.x + circle.w, circle.y + circle.w, circle.z, 0.0f },
			};
			hiprtFloat4* dAabbs = nullptr;
			CHECK_ORO( cudaMalloc( reinterpret_cast<void**>( &dAabbs ), sizeof( aabb ) ) );
			listCircles.aabbs = dAabbs;
			listCircles.aabbCount = 1;
			listCircles.aabbStride = 2 * sizeof( hiprtFloat4 );
			CHECK_ORO( cuMemcpyHtoD( reinterpret_cast<CUdeviceptr>( dAabbs ), aabb, sizeof( aabb ) ) );

			hiprtGeometryBuildInput geomInput{};
			geomInput.type = hiprtPrimitiveTypeAABBList;
			geomInput.primitive.aabbList = listCircles;
			geomInput.geomType = 1;

			hiprtBuildOptions options{};
			options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
			size_t geomTempSize = 0;
			hiprtDevicePtr geomTemp = nullptr;
			CHECK_HIPRT( hiprtGetGeometryBuildTemporaryBufferSize( ctxt, geomInput, options, geomTempSize ) );
			CHECK_ORO( cudaMalloc( reinterpret_cast<void**>( &geomTemp ), geomTempSize ) );
			CHECK_HIPRT( hiprtCreateGeometry( ctxt, geomInput, options, geoms[1] ) );
			CHECK_HIPRT( hiprtBuildGeometry( ctxt, hiprtBuildOperationBuild, geomInput, options, geomTemp, 0, geoms[1] ) );
			CHECK_ORO( cudaFree( geomTemp ) );
		}

		hiprtInstance instances[2]{};
		for ( int i = 0; i < 2; ++i )
		{
			instances[i].type = hiprtInstanceTypeGeometry;
			instances[i].geometry = geoms[i];
		}

		hiprtSceneBuildInput sceneInput{};
		sceneInput.instanceCount = 2;
		CHECK_ORO( cudaMalloc( reinterpret_cast<void**>( &sceneInput.instances ), sizeof( instances ) ) );
		CHECK_ORO( cuMemcpyHtoD( reinterpret_cast<CUdeviceptr>( sceneInput.instances ), instances, sizeof( instances ) ) );

		hiprtFrameSRT frames[2]{};
		frames[0].translation = { 0.0f, 0.1f, 0.0f };
		frames[0].scale = { 1.0f, 1.0f, 1.0f };
		frames[0].rotation = { 0.0f, 0.0f, 1.0f, 0.0f };
		frames[1].translation = { 0.0f, -0.1f, 0.0f };
		frames[1].scale = { 1.0f, 1.0f, 1.0f };
		frames[1].rotation = { 0.0f, 0.0f, 1.0f, 0.0f };
		sceneInput.frameCount = 2;
		sceneInput.frameType = hiprtFrameTypeSRT;
		CHECK_ORO( cudaMalloc( reinterpret_cast<void**>( &sceneInput.instanceFrames ), sizeof( frames ) ) );
		CHECK_ORO( cuMemcpyHtoD( reinterpret_cast<CUdeviceptr>( sceneInput.instanceFrames ), frames, sizeof( frames ) ) );

		hiprtBuildOptions sceneOptions{};
		sceneOptions.buildFlags = hiprtBuildFlagBitPreferFastBuild;
		size_t sceneTempSize = 0;
		hiprtDevicePtr sceneTemp = nullptr;
		CHECK_HIPRT( hiprtGetSceneBuildTemporaryBufferSize( ctxt, sceneInput, sceneOptions, sceneTempSize ) );
		CHECK_ORO( cudaMalloc( reinterpret_cast<void**>( &sceneTemp ), sceneTempSize ) );

		hiprtScene scene = nullptr;
		CHECK_HIPRT( hiprtCreateScene( ctxt, sceneInput, sceneOptions, scene ) );
		CHECK_HIPRT( hiprtBuildScene( ctxt, hiprtBuildOperationBuild, sceneInput, sceneOptions, sceneTemp, 0, scene ) );
		CHECK_ORO( cudaFree( sceneTemp ) );

		hiprtFuncNameSet funcNameSets[2]{};
		funcNameSets[0].intersectFuncName = "intersectSphere";
		funcNameSets[1].intersectFuncName = "intersectCircle";

		hiprtFloat4* dSphere = nullptr;
		hiprtFloat4* dCircle = nullptr;
		const hiprtFloat4 sphere = { -0.15f, 0.0f, 0.0f, 0.15f };
		const hiprtFloat4 circle = { 0.15f, 0.0f, 0.0f, 0.15f };
		CHECK_ORO( cudaMalloc( reinterpret_cast<void**>( &dSphere ), sizeof( sphere ) ) );
		CHECK_ORO( cudaMalloc( reinterpret_cast<void**>( &dCircle ), sizeof( circle ) ) );
		CHECK_ORO( cuMemcpyHtoD( reinterpret_cast<CUdeviceptr>( dSphere ), &sphere, sizeof( sphere ) ) );
		CHECK_ORO( cuMemcpyHtoD( reinterpret_cast<CUdeviceptr>( dCircle ), &circle, sizeof( circle ) ) );

		hiprtFuncDataSet funcDataSets[2]{};
		funcDataSets[0].intersectFuncData = dSphere;
		funcDataSets[1].intersectFuncData = dCircle;

		hiprtFuncTable funcTable;
		CHECK_HIPRT( hiprtCreateFuncTable( ctxt, 2, 1, funcTable ) );
		CHECK_HIPRT( hiprtSetFuncTable( ctxt, funcTable, 0, 0, funcDataSets[0] ) );
		CHECK_HIPRT( hiprtSetFuncTable( ctxt, funcTable, 1, 0, funcDataSets[1] ) );

		CUfunction func = nullptr;
		std::vector<hiprtFuncNameSet> funcNameSetVec = { funcNameSets[0], funcNameSets[1] };
		buildTraceKernel(
			ctxt,
			std::filesystem::path( HIPRTSDK_ROOT_DIR ) / "tutorials/common/MultiCustomTutorialKernels.h",
			"MultiCustomIntersectionKernel",
			func,
			nullptr,
			&funcNameSetVec,
			2,
			1 );

		uint8_t* pixels = nullptr;
		CHECK_ORO( cudaMalloc( reinterpret_cast<void**>( &pixels ), m_res.x * m_res.y * 4 ) );
		void* args[] = { &scene, &pixels, &funcTable, &m_res };
		launchKernel( func, m_res.x, m_res.y, args );
		writeImage( "11_multi_custom_intersection.png", m_res.x, m_res.y, pixels );

		CHECK_ORO( cudaFree( sceneInput.instances ) );
		CHECK_ORO( cudaFree( sceneInput.instanceFrames ) );
		CHECK_ORO( cudaFree( dSphere ) );
		CHECK_ORO( cudaFree( dCircle ) );
		CHECK_ORO( cudaFree( listSpheres.aabbs ) );
		CHECK_ORO( cudaFree( listCircles.aabbs ) );
		CHECK_ORO( cudaFree( pixels ) );
		CHECK_HIPRT( hiprtDestroyFuncTable( ctxt, funcTable ) );
		CHECK_HIPRT( hiprtDestroyGeometry( ctxt, geoms[0] ) );
		CHECK_HIPRT( hiprtDestroyGeometry( ctxt, geoms[1] ) );
		CHECK_HIPRT( hiprtDestroyScene( ctxt, scene ) );
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
