#include <hiprt/hiprt_device.h>
#include <hiprt/hiprt_math.h>
#include <hiprt/hiprt_vec.h>

HIPRT_DEVICE HIPRT_INLINE float3 gammaCorrect( float3 a )
{
	const float g = 1.0f / 2.2f;
	return { powf( a.x, g ), powf( a.y, g ), powf( a.z, g ) };
}

extern "C" __global__ void MotionBlurKernel( hiprtScene scene, uint8_t* pixels, hiprtInt2 res )
{
	const uint32_t x	 = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y	 = blockIdx.y * blockDim.y + threadIdx.y;
	if ( x >= static_cast<uint32_t>( res.x ) || y >= static_cast<uint32_t>( res.y ) ) return;

	const uint32_t index = x + y * res.x;
	constexpr uint32_t Samples = 32u;

	hiprtRay ray;
	ray.origin	  = { x / static_cast<float>( res.x ) - 0.5f, y / static_cast<float>( res.y ) - 0.5f, -1.0f };
	ray.direction = { 0.0f, 0.0f, 1.0f };

	const float3 colors[2] = { { 1.0f, 0.0f, 0.5f }, { 0.0f, 0.5f, 1.0f } };
	float3 color = { 0.0f, 0.0f, 0.0f };
	for ( uint32_t i = 0; i < Samples; ++i )
	{
		const float time = i / static_cast<float>( Samples );
		hiprtSceneTraversalClosest tr( scene, ray, hiprtFullRayMask, hiprtTraversalHintDefault, nullptr, nullptr, 0, time );
		const hiprtHit hit = tr.getNextHit();
		if ( hit.hasHit() )
		{
			const uint32_t instanceId = hit.instanceIDs[1] != hiprtInvalidValue ? hit.instanceIDs[1] : hit.instanceIDs[0];
			color += colors[instanceId];
		}
	}

	color				  = gammaCorrect( color / Samples );
	pixels[index * 4 + 0] = static_cast<uint8_t>( color.x * 255.0f );
	pixels[index * 4 + 1] = static_cast<uint8_t>( color.y * 255.0f );
	pixels[index * 4 + 2] = static_cast<uint8_t>( color.z * 255.0f );
	pixels[index * 4 + 3] = 255;
}
