//
// Copyright (c) 2021-2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//

#include <numeric>
#include <tutorials/common/TutorialBase.h>

class Tutorial : public TutorialBase
{
  public:
	void run()
	{
		constexpr uint32_t CircleCount = 3u;
		hiprtFloat4		   circles[] = { { 0.25f, 0.5f, 0.0f, 0.1f }, { 0.5f, 0.5f, 0.0f, 0.1f }, { 0.75f, 0.5f, 0.0f, 0.1f } };

		hiprtContext ctxt;
		CHECK_HIPRT( hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, ctxt ) );

		hiprtGeometryBuildInput geomBuildInputs[2];
		hiprtGeometry			geoms[2];

		hiprtAABBListPrimitive list;
		{
			list.aabbCount	= CircleCount;
			list.aabbStride = 2 * sizeof( hiprtFloat4 );
			hiprtFloat4 aabbs[2 * CircleCount];
			for ( int i = 0; i < CircleCount; i++ )
			{
				const hiprtFloat4& c = circles[i];
				aabbs[i * 2 + 0]	 = { c.x - c.w, c.y - c.w, c.z, 0.0f };
				aabbs[i * 2 + 1]	 = { c.x + c.w, c.y + c.w, c.z, 0.0f };
			}
			hiprtFloat4* dAabbs = nullptr;
			CHECK_ORO( cudaMalloc( reinterpret_cast<void**>( &dAabbs ), 2 * list.aabbCount * sizeof( hiprtFloat4 ) ) );
			list.aabbs = dAabbs;
			CHECK_ORO( cuMemcpyHtoD( reinterpret_cast<CUdeviceptr>( list.aabbs ), aabbs, 2 * list.aabbCount * sizeof( hiprtFloat4 ) ) );

			geomBuildInputs[0].type				  = hiprtPrimitiveTypeAABBList;
			geomBuildInputs[0].primitive.aabbList = list;
			geomBuildInputs[0].geomType			  = 0;
		}

		hiprtTriangleMeshPrimitive mesh;
		{
			mesh.triangleCount	= 3;
			mesh.triangleStride = sizeof( hiprtInt3 );
			std::vector<int> triangleIndices( 3 * mesh.triangleCount );
			std::iota( triangleIndices.begin(), triangleIndices.end(), 0 );
			hiprtInt3* dTriangleIndices = nullptr;
			CHECK_ORO( cudaMalloc( reinterpret_cast<void**>( &dTriangleIndices ), mesh.triangleCount * sizeof( hiprtInt3 ) ) );
			mesh.triangleIndices = dTriangleIndices;
			CHECK_ORO( cuMemcpyHtoD(
				reinterpret_cast<CUdeviceptr>( mesh.triangleIndices ),
				triangleIndices.data(),
				mesh.triangleCount * sizeof( hiprtInt3 ) ) );

			mesh.vertexCount	   = 9;
			mesh.vertexStride	   = sizeof( hiprtFloat3 );
			hiprtFloat3 vertices[] = {
				{ 0.15f, 0.40f, 0.0f },
				{ 0.35f, 0.40f, 0.0f },
				{ 0.25f, 0.60f, 0.0f },
				{ 0.40f, 0.40f, 0.0f },
				{ 0.60f, 0.40f, 0.0f },
				{ 0.50f, 0.60f, 0.0f },
				{ 0.65f, 0.40f, 0.0f },
				{ 0.85f, 0.40f, 0.0f },
				{ 0.75f, 0.60f, 0.0f } };
			hiprtFloat3* dVertices = nullptr;
			CHECK_ORO( cudaMalloc( reinterpret_cast<void**>( &dVertices ), mesh.vertexCount * sizeof( hiprtFloat3 ) ) );
			mesh.vertices = dVertices;
			CHECK_ORO( cuMemcpyHtoD(
				reinterpret_cast<CUdeviceptr>( mesh.vertices ), vertices, mesh.vertexCount * sizeof( hiprtFloat3 ) ) );

			geomBuildInputs[1].type					  = hiprtPrimitiveTypeTriangleMesh;
			geomBuildInputs[1].primitive.triangleMesh = mesh;
		}

		{
			hiprtBuildOptions options;
			options.buildFlags			   = hiprtBuildFlagBitPreferFastBuild;
			options.batchBuildMaxPrimCount = 16u; // this enables batch construction

			size_t		   geomTempSize;
			hiprtDevicePtr geomTemp;
			CHECK_HIPRT( hiprtGetGeometriesBuildTemporaryBufferSize( ctxt, 2, geomBuildInputs, options, geomTempSize ) );
			CHECK_ORO( cudaMalloc( reinterpret_cast<void**>( &geomTemp ), geomTempSize ) );

			hiprtGeometry* geomAddrs[] = { &geoms[0], &geoms[1] };
			CHECK_HIPRT( hiprtCreateGeometries( ctxt, 2, geomBuildInputs, options, geomAddrs ) );
			CHECK_HIPRT(
				hiprtBuildGeometries( ctxt, hiprtBuildOperationBuild, 2, geomBuildInputs, options, geomTemp, 0, geoms ) );
			CHECK_ORO( cudaFree( geomTemp ) );
		}

		hiprtScene			 scene;
		hiprtSceneBuildInput sceneInput;
		{
			hiprtInstance instance0;
			instance0.type	   = hiprtInstanceTypeGeometry;
			instance0.geometry = geoms[0];

			hiprtInstance instance1;
			instance1.type	   = hiprtInstanceTypeGeometry;
			instance1.geometry = geoms[1];

			hiprtInstance instances[] = { instance0, instance1 };

			sceneInput.instanceCount			= 2;
			sceneInput.instanceMasks			= nullptr;
			sceneInput.instanceTransformHeaders = nullptr;
			CHECK_ORO( cudaMalloc(
				reinterpret_cast<void**>( &sceneInput.instances ),
				sceneInput.instanceCount * sizeof( hiprtInstance ) ) );
			CHECK_ORO( cuMemcpyHtoD(
				reinterpret_cast<CUdeviceptr>( sceneInput.instances ),
				instances,
				sceneInput.instanceCount * sizeof( hiprtInstance ) ) );

			const float	  o = 0.05f;
			hiprtFrameSRT frames[2];
			frames[0].translation = { 0.0f, o, 0.0f };
			frames[0].scale		  = { 1.0f, 1.0f, 1.0f };
			frames[0].rotation	  = { 0.0f, 0.0f, 1.0f, 0.0f };
			frames[1].translation = { 0.0f, -o, 0.0f };
			frames[1].scale		  = { 1.0f, 1.0f, 1.0f };
			frames[1].rotation	  = { 0.0f, 0.0f, 1.0f, 0.0f };

			sceneInput.frameCount = 2;
			CHECK_ORO( cudaMalloc(
				reinterpret_cast<void**>( &sceneInput.instanceFrames ),
				sceneInput.frameCount * sizeof( hiprtFrameSRT ) ) );
			CHECK_ORO( cuMemcpyHtoD(
				reinterpret_cast<CUdeviceptr>( sceneInput.instanceFrames ),
				frames,
				sceneInput.frameCount * sizeof( hiprtFrameSRT ) ) );

			size_t			  sceneTempSize;
			hiprtDevicePtr	  sceneTemp;
			hiprtBuildOptions options;
			options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
			CHECK_HIPRT( hiprtGetSceneBuildTemporaryBufferSize( ctxt, sceneInput, options, sceneTempSize ) );
			CHECK_ORO( cudaMalloc( reinterpret_cast<void**>( &sceneTemp ), sceneTempSize ) );

			CHECK_HIPRT( hiprtCreateScene( ctxt, sceneInput, options, scene ) );
			CHECK_HIPRT( hiprtBuildScene( ctxt, hiprtBuildOperationBuild, sceneInput, options, sceneTemp, 0, scene ) );
			CHECK_ORO( cudaFree( sceneTemp ) );
		}

		hiprtFuncNameSet funcNameSet;
		funcNameSet.intersectFuncName			   = "intersectCircle";
		std::vector<hiprtFuncNameSet> funcNameSets = { funcNameSet };

		CUfunction func = nullptr;
		buildTraceKernel(
			ctxt,
			std::filesystem::path( HIPRTSDK_ROOT_DIR ) / "tutorials/common/SceneBuildTutorialKernels.h",
			"SceneBuildKernel",
			func,
			nullptr,
			&funcNameSets,
			1,
			1 );

		hiprtFuncDataSet funcDataSet;
		hiprtFloat4* dCircles = nullptr;
		CHECK_ORO( cudaMalloc( reinterpret_cast<void**>( &dCircles ), CircleCount * sizeof( hiprtFloat4 ) ) );
		funcDataSet.intersectFuncData = dCircles;
		CHECK_ORO( cuMemcpyHtoD( reinterpret_cast<CUdeviceptr>( dCircles ), circles, CircleCount * sizeof( hiprtFloat4 ) ) );

		hiprtFuncTable funcTable;
		CHECK_HIPRT( hiprtCreateFuncTable( ctxt, 1, 1, funcTable ) );
		CHECK_HIPRT( hiprtSetFuncTable( ctxt, funcTable, 0, 0, funcDataSet ) );

		uint8_t* pixels = nullptr;
		CHECK_ORO( cudaMalloc( reinterpret_cast<void**>( &pixels ), m_res.x * m_res.y * 4 ) );
		CHECK_ORO( cuMemsetD8( reinterpret_cast<CUdeviceptr>( pixels ), 0, m_res.x * m_res.y * 4 ) );

		void* args[] = { &scene, &pixels, &funcTable, &m_res };
		launchKernel( func, m_res.x, m_res.y, args );
		writeImage( "14_batch_build.png", m_res.x, m_res.y, pixels );

		CHECK_ORO( cudaFree( dCircles ) );
		CHECK_ORO( cudaFree( sceneInput.instances ) );
		CHECK_ORO( cudaFree( sceneInput.instanceFrames ) );
		CHECK_ORO( cudaFree( mesh.triangleIndices ) );
		CHECK_ORO( cudaFree( mesh.vertices ) );
		CHECK_ORO( cudaFree( list.aabbs ) );
		CHECK_ORO( cudaFree( pixels ) );

		CHECK_HIPRT( hiprtDestroyFuncTable( ctxt, funcTable ) );
		CHECK_HIPRT( hiprtDestroyGeometries( ctxt, 2, geoms ) );
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
