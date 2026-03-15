#include <hiprt/hiprt_device.h>
#include <hiprt/hiprt_math.h>
#include <hiprt/hiprt_vec.h>

__device__ bool intersectSphere( const hiprtRay& ray, const void* data, void* payload, hiprtHit& hit )
{
	const float4 sphere = reinterpret_cast<const float4*>( data )[hit.primID];
	const float3 center = hiprt::make_float3( sphere );
	const float	 radius = sphere.w;

	const float3 from = ray.origin;
	const float3 to	 = from + ray.direction * ray.maxT;
	const float3 m	 = from - center;
	const float3 d	 = to - from;
	const float  a	 = hiprt::dot( d, d );
	const float  b	 = 2.0f * hiprt::dot( m, d );
	const float  c	 = hiprt::dot( m, m ) - radius * radius;
	const float  dd	 = b * b - 4.0f * a * c;
	if ( dd < 0.0f ) return false;

	const float t = ( -b - sqrtf( dd ) ) / ( 2.0f * a );
	if ( t > 1.0f ) return false;

	hit.t		= t * ray.maxT;
	hit.normal	= hiprt::normalize( from + ray.direction * hit.t - center );
	return true;
}

__device__ bool intersectCircle( const hiprtRay& ray, const void* data, void* payload, hiprtHit& hit )
{
	const float4 circle = reinterpret_cast<const float4*>( data )[hit.primID];
	const float2 center = { circle.x, circle.y };
	const float  radius = circle.w;

	const float2 delta = { center.x - ray.origin.x, center.y - ray.origin.y };
	const float	 dist	= sqrtf( delta.x * delta.x + delta.y * delta.y );
	if ( dist >= radius ) return false;

	hit.normal = hiprt::normalize( float3{ dist, dist, dist } );
	return true;
}

extern "C" __global__ void
MultiCustomIntersectionKernel( hiprtScene scene, uint8_t* pixels, hiprtFuncTable table, hiprtInt2 res )
{
	const uint32_t x	 = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y	 = blockIdx.y * blockDim.y + threadIdx.y;
	if ( x >= static_cast<uint32_t>( res.x ) || y >= static_cast<uint32_t>( res.y ) ) return;

	const uint32_t index = x + y * res.x;
	hiprtRay ray;
	ray.origin	  = { x / static_cast<float>( res.x ) - 0.5f, y / static_cast<float>( res.y ) - 0.5f, -1.0f };
	ray.direction = { 0.0f, 0.0f, 1.0f };
	ray.maxT	  = 100000.0f;

	hiprtSceneTraversalClosest tr( scene, ray, hiprtFullRayMask, hiprtTraversalHintDefault, nullptr, table );
	const hiprtHit			   hit = tr.getNextHit();

	pixels[index * 4 + 0] = hit.hasHit() ? static_cast<uint8_t>( ( hit.normal.x + 1.0f ) * 0.5f * 255.0f ) : 0;
	pixels[index * 4 + 1] = hit.hasHit() ? static_cast<uint8_t>( ( hit.normal.y + 1.0f ) * 0.5f * 255.0f ) : 0;
	pixels[index * 4 + 2] = hit.hasHit() ? static_cast<uint8_t>( ( hit.normal.z + 1.0f ) * 0.5f * 255.0f ) : 0;
	pixels[index * 4 + 3] = 255;
}
