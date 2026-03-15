#include <hiprt/hiprt_device.h>
#include <hiprt/hiprt_math.h>
#include <hiprt/hiprt_vec.h>

extern "C" __global__ void GeomIntersectionKernel( hiprtGeometry geom, uint8_t* pixels, hiprtInt2 res )
{
	const uint32_t x	 = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y	 = blockIdx.y * blockDim.y + threadIdx.y;
	if ( x >= static_cast<uint32_t>( res.x ) || y >= static_cast<uint32_t>( res.y ) ) return;

	const uint32_t index = x + y * res.x;

	hiprtRay ray;
	ray.origin	  = { x / static_cast<float>( res.x ), y / static_cast<float>( res.y ), -1.0f };
	ray.direction = { 0.0f, 0.0f, 1.0f };

	hiprtGeomTraversalClosest tr( geom, ray );
	hiprtHit				  hit = tr.getNextHit();

	pixels[index * 4 + 0] = hit.hasHit() ? static_cast<uint8_t>( ( static_cast<float>( x ) / res.x ) * 255.0f ) : 0;
	pixels[index * 4 + 1] = hit.hasHit() ? static_cast<uint8_t>( ( static_cast<float>( y ) / res.y ) * 255.0f ) : 0;
	pixels[index * 4 + 2] = 0;
	pixels[index * 4 + 3] = 255;
}

extern "C" __global__ void SceneIntersectionKernel( hiprtScene scene, uint8_t* pixels, hiprtInt2 res )
{
	const uint32_t x	 = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y	 = blockIdx.y * blockDim.y + threadIdx.y;
	if ( x >= static_cast<uint32_t>( res.x ) || y >= static_cast<uint32_t>( res.y ) ) return;

	const uint32_t index = x + y * res.x;

	hiprtRay ray;
	ray.origin	  = { x / static_cast<float>( res.x ) - 0.5f, y / static_cast<float>( res.y ) - 0.5f, -1.0f };
	ray.direction = { 0.0f, 0.0f, 1.0f };

	hiprtSceneTraversalClosest tr( scene, ray, hiprtFullRayMask );
	hiprtHit				   hit = tr.getNextHit();

	pixels[index * 4 + 0] = hit.hasHit() ? static_cast<uint8_t>( ( static_cast<float>( x ) / res.x ) * 255.0f ) : 0;
	pixels[index * 4 + 1] = hit.hasHit() ? static_cast<uint8_t>( ( static_cast<float>( y ) / res.y ) * 255.0f ) : 0;
	pixels[index * 4 + 2] = 0;
	pixels[index * 4 + 3] = 255;
}
