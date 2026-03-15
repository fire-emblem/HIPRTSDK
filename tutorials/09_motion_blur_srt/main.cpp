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

		hiprtGeometry			   geoms[2]{};
		hiprtTriangleMeshPrimitive meshes[2]{};

		for ( int g = 0; g < 2; ++g )
		{
			meshes[g].triangleCount	 = 1;
			meshes[g].triangleStride = sizeof( hiprtInt3 );
			hiprtInt3* dTriangleIndices = nullptr;
			CHECK_ORO( cudaMalloc( reinterpret_cast<void**>( &dTriangleIndices ), sizeof( hiprtInt3 ) ) );
			meshes[g].triangleIndices = dTriangleIndices;
			const uint32_t triangleIndices[] = { 0, 1, 2 };
			CHECK_ORO( cuMemcpyHtoD( reinterpret_cast<CUdeviceptr>( dTriangleIndices ), triangleIndices, sizeof( hiprtInt3 ) ) );

			meshes[g].vertexCount  = 3;
			meshes[g].vertexStride = sizeof( hiprtFloat3 );
			hiprtFloat3* dVertices = nullptr;
			CHECK_ORO( cudaMalloc( reinterpret_cast<void**>( &dVertices ), 3 * sizeof( hiprtFloat3 ) ) );
			meshes[g].vertices = dVertices;
			constexpr float Scale = 0.15f;
			const hiprtFloat3 vertices[] = {
				{ Scale * sinf( 0.0f ), Scale * cosf( 0.0f ), 0.0f },
				{ Scale * sinf( hiprt::Pi * 2.0f / 3.0f ), Scale * cosf( hiprt::Pi * 2.0f / 3.0f ), 0.0f },
				{ Scale * sinf( hiprt::Pi * 4.0f / 3.0f ), Scale * cosf( hiprt::Pi * 4.0f / 3.0f ), 0.0f } };
			CHECK_ORO( cuMemcpyHtoD( reinterpret_cast<CUdeviceptr>( dVertices ), vertices, sizeof( vertices ) ) );

			hiprtGeometryBuildInput geomInput{};
			geomInput.type					 = hiprtPrimitiveTypeTriangleMesh;
			geomInput.primitive.triangleMesh = meshes[g];

			hiprtBuildOptions options{};
			options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
			size_t geomTempSize = 0;
			hiprtDevicePtr geomTemp = nullptr;
			CHECK_HIPRT( hiprtGetGeometryBuildTemporaryBufferSize( ctxt, geomInput, options, geomTempSize ) );
			CHECK_ORO( cudaMalloc( reinterpret_cast<void**>( &geomTemp ), geomTempSize ) );
			CHECK_HIPRT( hiprtCreateGeometry( ctxt, geomInput, options, geoms[g] ) );
			CHECK_HIPRT( hiprtBuildGeometry( ctxt, hiprtBuildOperationBuild, geomInput, options, geomTemp, 0, geoms[g] ) );
			CHECK_ORO( cudaFree( geomTemp ) );
		}

		hiprtInstance instances[2]{};
		for ( int i = 0; i < 2; ++i )
		{
			instances[i].type	  = hiprtInstanceTypeGeometry;
			instances[i].geometry = geoms[i];
		}

		hiprtSceneBuildInput sceneInput{};
		sceneInput.instanceCount = 2;
		CHECK_ORO( cudaMalloc( reinterpret_cast<void**>( &sceneInput.instances ), sizeof( instances ) ) );
		CHECK_ORO( cuMemcpyHtoD( reinterpret_cast<CUdeviceptr>( sceneInput.instances ), instances, sizeof( instances ) ) );

		constexpr float Offset = 0.3f;
		hiprtFrameSRT frames[5]{};
		frames[0].translation = { -0.25f, -Offset, 0.0f };
		frames[0].scale	   = { 1.0f, 1.0f, 1.0f };
		frames[0].rotation	   = { 0.0f, 0.0f, 1.0f, 0.0f };
		frames[0].time		   = 0.0f;
		frames[1].translation = { 0.0f, -Offset, 0.0f };
		frames[1].scale	   = { 1.0f, 1.0f, 1.0f };
		frames[1].rotation	   = { 0.0f, 0.0f, 1.0f, 0.0f };
		frames[1].time		   = 0.35f;
		frames[2].translation = { 0.25f, -Offset, 0.0f };
		frames[2].scale	   = { 1.0f, 1.0f, 1.0f };
		frames[2].rotation	   = { 0.0f, 0.0f, 1.0f, hiprt::Pi * 0.25f };
		frames[2].time		   = 1.0f;
		frames[3].translation = { 0.0f, Offset, 0.0f };
		frames[3].scale	   = { 1.0f, 1.0f, 1.0f };
		frames[3].rotation	   = { 0.0f, 0.0f, 1.0f, 0.0f };
		frames[3].time		   = 0.0f;
		frames[4].translation = { 0.0f, Offset, 0.0f };
		frames[4].scale	   = { 1.0f, 1.0f, 1.0f };
		frames[4].rotation	   = { 0.0f, 0.0f, 1.0f, hiprt::Pi * 0.5f };
		frames[4].time		   = 1.0f;

		sceneInput.frameCount = 5;
		sceneInput.frameType	 = hiprtFrameTypeSRT;
		CHECK_ORO( cudaMalloc( reinterpret_cast<void**>( &sceneInput.instanceFrames ), sizeof( frames ) ) );
		CHECK_ORO( cuMemcpyHtoD( reinterpret_cast<CUdeviceptr>( sceneInput.instanceFrames ), frames, sizeof( frames ) ) );

		hiprtTransformHeader headers[2]{};
		headers[0].frameIndex = 0;
		headers[0].frameCount = 3;
		headers[1].frameIndex = 3;
		headers[1].frameCount = 2;
		CHECK_ORO( cudaMalloc( reinterpret_cast<void**>( &sceneInput.instanceTransformHeaders ), sizeof( headers ) ) );
		CHECK_ORO( cuMemcpyHtoD(
			reinterpret_cast<CUdeviceptr>( sceneInput.instanceTransformHeaders ), headers, sizeof( headers ) ) );

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

		CUfunction func = nullptr;
		buildTraceKernel(
			ctxt,
			std::filesystem::path( HIPRTSDK_ROOT_DIR ) / "tutorials/common/MotionBlurTutorialKernels.h",
			"MotionBlurKernel",
			func );

		uint8_t* pixels = nullptr;
		CHECK_ORO( cudaMalloc( reinterpret_cast<void**>( &pixels ), m_res.x * m_res.y * 4 ) );
		void* args[] = { &scene, &pixels, &m_res };
		launchKernel( func, m_res.x, m_res.y, args );
		writeImage( "09_motion_blur_srt.png", m_res.x, m_res.y, pixels );

		CHECK_ORO( cudaFree( sceneInput.instances ) );
		CHECK_ORO( cudaFree( sceneInput.instanceFrames ) );
		CHECK_ORO( cudaFree( sceneInput.instanceTransformHeaders ) );
		CHECK_ORO( cudaFree( pixels ) );
		for ( int g = 0; g < 2; ++g )
		{
			CHECK_ORO( cudaFree( meshes[g].vertices ) );
			CHECK_ORO( cudaFree( meshes[g].triangleIndices ) );
			CHECK_HIPRT( hiprtDestroyGeometry( ctxt, geoms[g] ) );
		}
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
