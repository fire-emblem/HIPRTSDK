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

#pragma once
#include <array>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <filesystem>
#include <fstream>
#include <map>
#include <hiprt/hiprt.h>
#include <nvrtc.h>
#include <optional>
#include <string>
#include <tutorials/common/Common.h>
#include <vector>

#define CHECK_ORO( error ) ( checkOro( error, __FILE__, __LINE__ ) )
void checkOro( cudaError_t res, const char* file, uint32_t line );
void checkOro( CUresult res, const char* file, uint32_t line );

#define CHECK_HIPRT( error ) ( checkHiprt( error, __FILE__, __LINE__ ) )
void checkHiprt( hiprtError res, const char* file, uint32_t line );

#define CHECK_NVRTC( error ) ( checkNvrtc( error, __FILE__, __LINE__ ) )
void checkNvrtc( nvrtcResult res, const char* file, uint32_t line );

class TutorialBase
{
  public:
	virtual ~TutorialBase() { printf( "success\n" ); }
	void init( uint32_t deviceIndex = 0 );

	virtual void run() = 0;

	void buildTraceKernel(
		hiprtContext				   ctxt,
		const std::filesystem::path&   path,
		const std::string&			   functionName,
		CUfunction&					   functionOut,
		std::vector<const char*>*	   opts			= nullptr,
		std::vector<hiprtFuncNameSet>* funcNameSets = nullptr,
		uint32_t					   numGeomTypes = 0,
		uint32_t					   numRayTypes	= 1 );

	void launchKernel( CUfunction func, uint32_t nx, uint32_t ny, void** args );
	void launchKernel( CUfunction func, uint32_t nx, uint32_t ny, uint32_t bx, uint32_t by, void** args );

	static void writeImage( const std::string& path, uint32_t width, uint32_t height, uint8_t* pixels );

	static bool readSourceCode(
		const std::filesystem::path&					  path,
		std::string&									  sourceCode,
		std::optional<std::vector<std::filesystem::path>> includes = std::nullopt );

  protected:
	hiprtContextCreationInput m_ctxtInput;
	CUcontext				  m_cudaCtx;
	CUdevice				  m_cudaDevice;
	hiprtInt2				  m_res;
	std::map<std::string, CUmodule> m_moduleCache;
};
