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
#include <tutorials/common/BvhBuilder.h>
#include <tutorials/common/CornellBox.h>
#include <tutorials/common/TutorialBase.h>

class Tutorial : public TutorialBase
{
  public:
	void buildBvh( hiprtGeometryBuildInput& buildInput );
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
		buildBvh( geomInput );

		size_t			  geomTempSize;
		hiprtDevicePtr	  geomTemp;
		hiprtBuildOptions options;
		options.buildFlags = hiprtBuildFlagBitCustomBvhImport;
		CHECK_HIPRT( hiprtGetGeometryBuildTemporaryBufferSize( ctxt, geomInput, options, geomTempSize ) );
		CHECK_ORO( cudaMalloc( reinterpret_cast<void**>( &geomTemp ), geomTempSize ) );

		hiprtGeometry geom;
		CHECK_HIPRT( hiprtCreateGeometry( ctxt, geomInput, options, geom ) );
		CHECK_HIPRT( hiprtBuildGeometry( ctxt, hiprtBuildOperationBuild, geomInput, options, geomTemp, 0, geom ) );

		CUfunction func = nullptr;
		buildTraceKernel(
			ctxt,
			std::filesystem::path( HIPRTSDK_ROOT_DIR ) / "tutorials/common/CustomBvhImportTutorialKernels.h",
			"CustomBvhImportKernel",
			func );

		uint8_t* pixels = nullptr;
		CHECK_ORO( cudaMalloc( reinterpret_cast<void**>( &pixels ), m_res.x * m_res.y * 4 ) );

		uint32_t* matIndices = nullptr;
		CHECK_ORO( cudaMalloc( reinterpret_cast<void**>( &matIndices ), mesh.triangleCount * sizeof( uint32_t ) ) );
		CHECK_ORO( cuMemcpyHtoD(
			reinterpret_cast<CUdeviceptr>( matIndices ),
			cornellBoxMatIndices.data(),
			mesh.triangleCount * sizeof( uint32_t ) ) );

		hiprtFloat3* diffusColors = nullptr;
		CHECK_ORO(
			cudaMalloc( reinterpret_cast<void**>( &diffusColors ), CornellBoxMaterialCount * sizeof( hiprtFloat3 ) ) );
		CHECK_ORO( cuMemcpyHtoD(
			reinterpret_cast<CUdeviceptr>( diffusColors ),
			const_cast<hiprtFloat3*>( cornellBoxDiffuseColors.data() ),
			CornellBoxMaterialCount * sizeof( hiprtFloat3 ) ) );

		void* args[] = { &geom, &pixels, &m_res, &matIndices, &diffusColors };
		launchKernel( func, m_res.x, m_res.y, args );
		writeImage( "07_custom_bvh_import.png", m_res.x, m_res.y, pixels );

		CHECK_ORO( cudaFree( matIndices ) );
		CHECK_ORO( cudaFree( diffusColors ) );
		CHECK_ORO( cudaFree( dTriangleIndices ) );
		CHECK_ORO( cudaFree( dVertices ) );
		CHECK_ORO( cudaFree( pixels ) );
		CHECK_ORO( cudaFree( geomTemp ) );
		CHECK_ORO( cudaFree( geomInput.nodeList.leafNodes ) );
		CHECK_ORO( cudaFree( geomInput.nodeList.internalNodes ) );

		CHECK_HIPRT( hiprtDestroyGeometry( ctxt, geom ) );
		CHECK_HIPRT( hiprtDestroyContext( ctxt ) );
	}
};

void Tutorial::buildBvh( hiprtGeometryBuildInput& buildInput )
{
	std::vector<hiprtInternalNode> internalNodes;
	std::vector<Aabb>			   primBoxes;
	if ( buildInput.type == hiprtPrimitiveTypeTriangleMesh )
	{
		primBoxes.resize( buildInput.primitive.triangleMesh.triangleCount );
		std::vector<uint8_t> verticesRaw(
			buildInput.primitive.triangleMesh.vertexCount * buildInput.primitive.triangleMesh.vertexStride );
		std::vector<uint8_t> trianglesRaw(
			buildInput.primitive.triangleMesh.triangleCount * buildInput.primitive.triangleMesh.triangleStride );
		CHECK_ORO( cuMemcpyDtoH(
			verticesRaw.data(),
			reinterpret_cast<CUdeviceptr>( buildInput.primitive.triangleMesh.vertices ),
			buildInput.primitive.triangleMesh.vertexCount * buildInput.primitive.triangleMesh.vertexStride ) );
		CHECK_ORO( cuMemcpyDtoH(
			trianglesRaw.data(),
			reinterpret_cast<CUdeviceptr>( buildInput.primitive.triangleMesh.triangleIndices ),
			buildInput.primitive.triangleMesh.triangleCount * buildInput.primitive.triangleMesh.triangleStride ) );
		for ( uint32_t i = 0; i < buildInput.primitive.triangleMesh.triangleCount; ++i )
		{
			hiprtInt3 triangle =
				*reinterpret_cast<hiprtInt3*>( trianglesRaw.data() + i * buildInput.primitive.triangleMesh.triangleStride );
			hiprtFloat3 v0 = *reinterpret_cast<const hiprtFloat3*>(
				verticesRaw.data() + triangle.x * buildInput.primitive.triangleMesh.vertexStride );
			hiprtFloat3 v1 = *reinterpret_cast<const hiprtFloat3*>(
				verticesRaw.data() + triangle.y * buildInput.primitive.triangleMesh.vertexStride );
			hiprtFloat3 v2 = *reinterpret_cast<const hiprtFloat3*>(
				verticesRaw.data() + triangle.z * buildInput.primitive.triangleMesh.vertexStride );
			primBoxes[i].reset();
			primBoxes[i].grow( v0 );
			primBoxes[i].grow( v1 );
			primBoxes[i].grow( v2 );
		}
		BvhBuilder::build( buildInput.primitive.triangleMesh.triangleCount, primBoxes, internalNodes );
	}
	else if ( buildInput.type == hiprtPrimitiveTypeAABBList )
	{
		primBoxes.resize( buildInput.primitive.aabbList.aabbCount );
		std::vector<uint8_t> primBoxesRaw( buildInput.primitive.aabbList.aabbCount * buildInput.primitive.aabbList.aabbStride );
		CHECK_ORO( cuMemcpyDtoH(
			primBoxesRaw.data(),
			reinterpret_cast<CUdeviceptr>( buildInput.primitive.aabbList.aabbs ),
			buildInput.primitive.aabbList.aabbCount * buildInput.primitive.aabbList.aabbStride ) );
		for ( uint32_t i = 0; i < buildInput.primitive.aabbList.aabbCount; ++i )
		{
			hiprtFloat4* ptr = reinterpret_cast<hiprtFloat4*>( primBoxesRaw.data() + i * buildInput.primitive.aabbList.aabbStride );
			primBoxes[i].m_min = { ptr[0].x, ptr[0].y, ptr[0].z };
			primBoxes[i].m_max = { ptr[1].x, ptr[1].y, ptr[1].z };
		}
		BvhBuilder::build( buildInput.primitive.aabbList.aabbCount, primBoxes, internalNodes );
	}

	std::vector<hiprtLeafNode> leafNodes( primBoxes.size() );
	for ( uint32_t i = 0; i < primBoxes.size(); ++i )
	{
		leafNodes[i].primID	 = i;
		leafNodes[i].aabbMin = primBoxes[i].m_min;
		leafNodes[i].aabbMax = primBoxes[i].m_max;
	}

	buildInput.nodeList.nodeCount = static_cast<uint32_t>( leafNodes.size() );

	CHECK_ORO( cudaMalloc( reinterpret_cast<void**>( &buildInput.nodeList.leafNodes ), leafNodes.size() * sizeof( hiprtLeafNode ) ) );
	CHECK_ORO( cuMemcpyHtoD(
		reinterpret_cast<CUdeviceptr>( buildInput.nodeList.leafNodes ),
		leafNodes.data(),
		leafNodes.size() * sizeof( hiprtLeafNode ) ) );
	CHECK_ORO( cudaMalloc(
		reinterpret_cast<void**>( &buildInput.nodeList.internalNodes ),
		internalNodes.size() * sizeof( hiprtInternalNode ) ) );
	CHECK_ORO( cuMemcpyHtoD(
		reinterpret_cast<CUdeviceptr>( buildInput.nodeList.internalNodes ),
		internalNodes.data(),
		internalNodes.size() * sizeof( hiprtInternalNode ) ) );
}

int main( int argc, char** argv )
{
	Tutorial tutorial;
	tutorial.init( 0 );
	tutorial.run();

	return 0;
}
