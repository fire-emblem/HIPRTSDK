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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//

#include <tutorials/common/TutorialBase.h>

int main( int argc, char** argv )
{
	const int deviceIndex = 0;

	CHECK_ORO( cuInit( 0 ) );

	CUdevice cudaDevice;
	CHECK_ORO( cuDeviceGet( &cudaDevice, deviceIndex ) );

	CUcontext cudaCtx;
	CHECK_ORO( cuCtxCreate( &cudaCtx, 0, cudaDevice ) );

	cudaDeviceProp props;
	CHECK_ORO( cudaGetDeviceProperties( &props, deviceIndex ) );
	std::cout << "Executing on '" << props.name << "'" << std::endl;

	hiprtContextCreationInput ctxtInput{};
	ctxtInput.deviceType = hiprtDeviceNVIDIA;
	ctxtInput.ctxt	   = cudaCtx;
	ctxtInput.device	   = cudaDevice;

	hiprtContext ctxt;
	CHECK_HIPRT( hiprtCreateContext( HIPRT_API_VERSION, ctxtInput, ctxt ) );
	CHECK_HIPRT( hiprtDestroyContext( ctxt ) );
	CHECK_ORO( cuCtxDestroy( cudaCtx ) );

	return 0;
}
