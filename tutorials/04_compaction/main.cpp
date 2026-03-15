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
#include <tutorials/common/CornellBox.h>
#include <tutorials/common/TutorialBase.h>

class Tutorial : public TutorialBase
{
  public:
	void run()
	{
		hiprtContext ctxt;
		CHECK_HIPRT( hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, ctxt ) );

		hiprtTriangleMeshPrimitive mesh;
		mesh.triangleCount	= CornellBoxTriangleCount;
		mesh.triangleStride = sizeof( hiprtInt3 );
		std::array<uint32_t, 3 * CornellBoxTriangleCount> triangleIndices;
		std::iota( triangleIndices.begin(), triangleIndices.end(), 0 );
		hiprtInt3* dTriangleIndices = nullptr;
		CHECK_ORO(
			cudaMalloc( reinterpret_cast<void**>( &dTriangleIndices ), mesh.triangleCount * sizeof( hiprtInt3 ) ) );
		mesh.triangleIndices = dTriangleIndices;
		CHECK_ORO( cuMemcpyHtoD(
			reinterpret_cast<CUdeviceptr>( mesh.triangleIndices ),
			triangleIndices.data(),
			mesh.triangleCount * sizeof( hiprtInt3 ) ) );

		mesh.vertexCount  = 3 * mesh.triangleCount;
		mesh.vertexStride = sizeof( hiprtFloat3 );
		hiprtFloat3* dVertices = nullptr;
		CHECK_ORO( cudaMalloc( reinterpret_cast<void**>( &dVertices ), mesh.vertexCount * sizeof( hiprtFloat3 ) ) );
		mesh.vertices = dVertices;
		CHECK_ORO( cuMemcpyHtoD(
			reinterpret_cast<CUdeviceptr>( mesh.vertices ),
			const_cast<hiprtFloat3*>( cornellBoxVertices.data() ),
			mesh.vertexCount * sizeof( hiprtFloat3 ) ) );

		hiprtGeometryBuildInput geomInput;
		geomInput.type					 = hiprtPrimitiveTypeTriangleMesh;
		geomInput.primitive.triangleMesh = mesh;

		size_t			  geomTempSize;
		hiprtDevicePtr	  geomTemp;
		hiprtBuildOptions options;
		options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
		CHECK_HIPRT( hiprtGetGeometryBuildTemporaryBufferSize( ctxt, geomInput, options, geomTempSize ) );
		CHECK_ORO( cudaMalloc( reinterpret_cast<void**>( &geomTemp ), geomTempSize ) );

		hiprtGeometry geom;
		CHECK_HIPRT( hiprtCreateGeometry( ctxt, geomInput, options, geom ) );
		CHECK_HIPRT( hiprtBuildGeometry( ctxt, hiprtBuildOperationBuild, geomInput, options, geomTemp, 0, geom ) );
		CHECK_HIPRT( hiprtCompactGeometry( ctxt, 0, geom, geom ) );

		CUfunction func = nullptr;
		buildTraceKernel(
			ctxt, std::filesystem::path( HIPRTSDK_ROOT_DIR ) / "tutorials/common/CornellTutorialKernels.h", "CornellBoxKernel", func );

		uint8_t* pixels = nullptr;
		CHECK_ORO( cudaMalloc( reinterpret_cast<void**>( &pixels ), m_res.x * m_res.y * 4 ) );

		void* args[] = { &geom, &pixels, &m_res };
		launchKernel( func, m_res.x, m_res.y, args );
		writeImage( "04_compaction.png", m_res.x, m_res.y, pixels );

		CHECK_ORO( cudaFree( dTriangleIndices ) );
		CHECK_ORO( cudaFree( dVertices ) );
		CHECK_ORO( cudaFree( geomTemp ) );
		CHECK_ORO( cudaFree( pixels ) );

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
