#include <hiprt/hiprt_device.h>
#include <hiprt/hiprt_math.h>
#include <hiprt/hiprt_vec.h>

__device__ bool intersectCircle( const hiprtRay& ray, const void* data, void* payload, hiprtHit& hit )
{
	const float4* circles = reinterpret_cast<const float4*>( data );
	const float4  circle  = circles[hit.primID];
	const float2  center  = { circle.x, circle.y };
	const float   radius  = circle.w;

	const float2 delta = { center.x - ray.origin.x, center.y - ray.origin.y };
	const float	 dist	= sqrtf( delta.x * delta.x + delta.y * delta.y );
	if ( dist >= radius ) return false;

	hit.normal = hiprt::normalize( float3{ dist, dist, dist } );
	return true;
}

extern "C" __global__ void SceneBuildKernel( hiprtScene scene, uint8_t* pixels, hiprtFuncTable table, hiprtInt2 res )
{
	const uint32_t x	 = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y	 = blockIdx.y * blockDim.y + threadIdx.y;
	if ( x >= static_cast<uint32_t>( res.x ) || y >= static_cast<uint32_t>( res.y ) ) return;

	const uint32_t index = x + y * res.x;

	hiprtRay ray;
	ray.origin	  = { x / static_cast<float>( res.x ), y / static_cast<float>( res.y ), -1.0f };
	ray.direction = { 0.0f, 0.0f, 1.0f };
	ray.maxT	  = 1000.0f;

	const float3 colors[2][3] = {
		{ { 1.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 0.0f }, { 0.0f, 0.0f, 1.0f } },
		{ { 0.0f, 1.0f, 1.0f }, { 1.0f, 0.0f, 1.0f }, { 1.0f, 1.0f, 0.0f } },
	};

	hiprtSceneTraversalAnyHit tr( scene, ray, hiprtFullRayMask, hiprtTraversalHintDefault, nullptr, table );
	while ( true )
	{
		const hiprtHit hit = tr.getNextHit();
		int3		   color = { 0, 0, 0 };
		if ( hit.hasHit() )
		{
			const uint32_t instanceID = hit.instanceIDs[1] != hiprtInvalidValue ? hit.instanceIDs[1] : hit.instanceIDs[0];
			const float3   diffuseColor = colors[instanceID][hit.primID];
			color.x = static_cast<int>( diffuseColor.x * 255.0f );
			color.y = static_cast<int>( diffuseColor.y * 255.0f );
			color.z = static_cast<int>( diffuseColor.z * 255.0f );
		}

		pixels[index * 4 + 0] += color.x;
		pixels[index * 4 + 1] += color.y;
		pixels[index * 4 + 2] += color.z;
		pixels[index * 4 + 3] = 255;

		if ( tr.getCurrentState() == hiprtTraversalStateFinished ) break;
	}
}
