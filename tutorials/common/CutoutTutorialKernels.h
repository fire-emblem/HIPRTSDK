#include <hiprt/hiprt_device.h>
#include <hiprt/hiprt_math.h>
#include <hiprt/hiprt_vec.h>

HIPRT_DEVICE HIPRT_INLINE bool cutoutFilter( const hiprtRay& ray, const void* data, void* payload, const hiprtHit& hit )
{
	constexpr float scale = 16.0f;
	const float2&	uv	  = hit.uv;
	float2			texCoord[2];
	texCoord[0] = ( 1.0f - uv.x - uv.y ) * float2{ 0.0f, 0.0f } + uv.x * float2{ 0.0f, 1.0f } + uv.y * float2{ 1.0f, 1.0f };
	texCoord[1] = ( 1.0f - uv.x - uv.y ) * float2{ 0.0f, 0.0f } + uv.x * float2{ 1.0f, 1.0f } + uv.y * float2{ 1.0f, 0.0f };
	return ( static_cast<uint32_t>( scale * texCoord[hit.primID].x ) + static_cast<uint32_t>( scale * texCoord[hit.primID].y ) ) & 1;
}

extern "C" __global__ void CutoutKernel( hiprtGeometry geom, uint8_t* pixels, hiprtFuncTable table, hiprtInt2 res )
{
	const uint32_t x	 = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y	 = blockIdx.y * blockDim.y + threadIdx.y;
	if ( x >= static_cast<uint32_t>( res.x ) || y >= static_cast<uint32_t>( res.y ) ) return;

	const uint32_t index = x + y * res.x;

	hiprtRay ray;
	ray.origin	  = { x / static_cast<float>( res.x ), y / static_cast<float>( res.y ), -1.0f };
	ray.direction = { 0.0f, 0.0f, 1.0f };

	hiprtGeomTraversalClosest tr( geom, ray, hiprtTraversalHintDefault, nullptr, table );
	hiprtHit				  hit = tr.getNextHit();

	pixels[index * 4 + 0] = hit.hasHit() ? 255 : 0;
	pixels[index * 4 + 1] = hit.hasHit() ? 255 : 0;
	pixels[index * 4 + 2] = hit.hasHit() ? 255 : 0;
	pixels[index * 4 + 3] = 255;
}
