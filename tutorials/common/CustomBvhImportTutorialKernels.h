#include <hiprt/hiprt_device.h>
#include <hiprt/hiprt_math.h>
#include <hiprt/hiprt_vec.h>

extern "C" __global__ void
CustomBvhImportKernel( hiprtGeometry geom, uint8_t* pixels, hiprtInt2 res, int* matIndices, float3* diffusColors )
{
	const uint32_t x	 = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y	 = blockIdx.y * blockDim.y + threadIdx.y;
	if ( x >= static_cast<uint32_t>( res.x ) || y >= static_cast<uint32_t>( res.y ) ) return;

	const uint32_t index = x + y * res.x;

	hiprtRay ray;
	ray.origin = { 278.0f, 273.0f, -900.0f };
	const float2 d = { 2.0f * x / static_cast<float>( res.x ) - 1.0f, 2.0f * y / static_cast<float>( res.y ) - 1.0f };
	const float3 uvw = { -387.817566f, -387.817566f, 1230.0f };
	ray.direction = { uvw.x * d.x, uvw.y * d.y, uvw.z };
	ray.direction =
		ray.direction /
		sqrtf( ray.direction.x * ray.direction.x + ray.direction.y * ray.direction.y + ray.direction.z * ray.direction.z );

	hiprtGeomTraversalAnyHit tr( geom, ray, hiprtTraversalHintDefault );
	while ( tr.getCurrentState() != hiprtTraversalStateFinished )
	{
		const hiprtHit hit = tr.getNextHit();
		int3		   color = { 0, 0, 0 };
		if ( hit.hasHit() )
		{
			const int	  matIndex	   = matIndices[hit.primID];
			const float	  alpha		   = 1.0f / 3.0f;
			const float3 diffuseColor = alpha * diffusColors[matIndex];
			color.x				   = static_cast<int>( diffuseColor.x * 255.0f );
			color.y				   = static_cast<int>( diffuseColor.y * 255.0f );
			color.z				   = static_cast<int>( diffuseColor.z * 255.0f );
		}

		pixels[index * 4 + 0] = min( 255, static_cast<int>( pixels[index * 4 + 0] ) + color.x );
		pixels[index * 4 + 1] = min( 255, static_cast<int>( pixels[index * 4 + 1] ) + color.y );
		pixels[index * 4 + 2] = min( 255, static_cast<int>( pixels[index * 4 + 2] ) + color.z );
		pixels[index * 4 + 3] = 255;
	}
}
