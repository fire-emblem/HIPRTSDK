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

#include <tutorials/common/TutorialBase.h>

#include <cstdlib>
#include <sstream>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <contrib/stbi/stbi_image_write.h>

namespace
{
std::string quoteShellArg( const std::string& arg )
{
	std::string escaped = "'";
	for ( const char ch : arg )
	{
		if ( ch == '\'' ) escaped += "'\\''";
		else escaped += ch;
	}
	escaped += "'";
	return escaped;
}

std::string findCudaCompiler()
{
	const std::vector<std::string> candidates = {
		std::getenv( "CUDACXX" ) ? std::getenv( "CUDACXX" ) : "",
		std::getenv( "CUDA_PATH" ) ? std::string( std::getenv( "CUDA_PATH" ) ) + "/bin/nvcc" : "",
		"/opt/maca/tools/cu-bridge/bin/cucc",
		"/opt/maca/tools/cu-bridge/CUDA_DIR/bin/nvcc",
		"/root/cu-bridge/CUDA_DIR/bin/nvcc",
		"nvcc",
	};

	for ( const auto& candidate : candidates )
	{
		if ( candidate.empty() ) continue;
		if ( candidate == "nvcc" ) return candidate;
		if ( std::filesystem::exists( candidate ) ) return candidate;
	}
	return {};
}
} // namespace

void checkOro( cudaError_t res, const char* file, uint32_t line )
{
	if ( res != cudaSuccess )
	{
		const char* msg = cudaGetErrorString( res );
		std::cerr << "Orochi error: '" << msg << "' on line " << line << " "
				  << " in '" << file << "'." << std::endl;
		exit( EXIT_FAILURE );
	}
}

void checkOro( CUresult res, const char* file, uint32_t line )
{
	if ( res != CUDA_SUCCESS )
	{
		const char* msg = nullptr;
		cuGetErrorString( res, &msg );
		std::cerr << "CUDA error: '" << ( msg ? msg : "unknown" ) << "' [ " << res << " ] on line " << line << " "
				  << " in '" << file << "'." << std::endl;
		exit( EXIT_FAILURE );
	}
}

void checkNvrtc( nvrtcResult res, const char* file, uint32_t line )
{
	if ( res != NVRTC_SUCCESS )
	{
		std::cerr << "NVRTC error: '" << nvrtcGetErrorString( res ) << "' [ " << res << " ] on line " << line << " "
				  << " in '" << file << "'." << std::endl;
		exit( EXIT_FAILURE );
	}
}

void checkHiprt( hiprtError res, const char* file, uint32_t line )
{
	if ( res != hiprtSuccess )
	{
		std::cerr << "HIPRT error: '" << res << "' on line " << line << " "
				  << " in '" << file << "'." << std::endl;
		exit( EXIT_FAILURE );
	}
}

void TutorialBase::init( uint32_t deviceIndex )
{
	m_res = { 512, 512 };

	CHECK_ORO( cuInit( 0 ) );
	CHECK_ORO( cuDeviceGet( &m_cudaDevice, deviceIndex ) );
	CHECK_ORO( cuCtxCreate( &m_cudaCtx, 0, m_cudaDevice ) );

	cudaDeviceProp props{};
	CHECK_ORO( cudaGetDeviceProperties( &props, deviceIndex ) );

	std::cout << "hiprt ver." << HIPRT_VERSION_STR << std::endl;
	std::cout << "Executing on '" << props.name << "'" << std::endl;
	m_ctxtInput.deviceType = hiprtDeviceNVIDIA;
	m_ctxtInput.ctxt	   = m_cudaCtx;
	m_ctxtInput.device	   = m_cudaDevice;
}

bool TutorialBase::readSourceCode(
	const std::filesystem::path& path, std::string& sourceCode, std::optional<std::vector<std::filesystem::path>> includes )
{
	std::fstream f( path );
	if ( f.is_open() )
	{
		size_t sizeFile;
		f.seekg( 0, std::fstream::end );
		size_t size = sizeFile = static_cast<size_t>( f.tellg() );
		f.seekg( 0, std::fstream::beg );
		if ( includes )
		{
			sourceCode.clear();
			std::string line;
				while ( std::getline( f, line ) )
				{
					if ( line.find( "#include" ) != std::string::npos )
					{
						size_t		pa	= line.find( "<" );
						size_t		pb	= line.find( ">" );
						std::string buf = line.substr( pa + 1, pb - pa - 1 );
						includes.value().push_back( buf );
					}
					sourceCode += line + '\n';
				}
		}
		else
		{
			sourceCode.resize( size, ' ' );
			f.read( &sourceCode[0], size );
		}
		f.close();
	}
	else
		return false;
	return true;
}

void TutorialBase::buildTraceKernel(
	hiprtContext				   ctxt,
	const std::filesystem::path&   path,
	const std::string&			   functionName,
	CUfunction&					   functionOut,
	std::vector<const char*>*	   opts,
	std::vector<hiprtFuncNameSet>* funcNameSets,
	uint32_t					   numGeomTypes,
	uint32_t					   numRayTypes )
{
	std::vector<const char*>		   options;
	std::vector<std::filesystem::path> includeNamesData;
	std::string						   sourceCode;

	if ( !readSourceCode( path, sourceCode, includeNamesData ) )
	{
		std::cerr << "Unable to find file '" << path << "'" << std::endl;
		exit( EXIT_FAILURE );
	}

	std::vector<std::string> headersData( includeNamesData.size() );
	std::vector<std::string> includeNameStrings( includeNamesData.size() );
	std::vector<const char*> headers;
	std::vector<const char*> includeNames;
	const auto sdkRoot = std::filesystem::path( HIPRTSDK_ROOT_DIR );
	for ( size_t i = 0; i < includeNamesData.size(); i++ )
	{
		const std::vector<std::filesystem::path> candidates = {
			sdkRoot / "tutorials" / includeNamesData[i],
			sdkRoot / includeNamesData[i],
			std::filesystem::path( HIPRT_ROOT_DIRECTORY ) / includeNamesData[i],
		};

		bool found = false;
		for ( const auto& candidate : candidates )
		{
			if ( readSourceCode( candidate, headersData[i] ) )
			{
				found = true;
				break;
			}
		}
		if ( !found )
		{
			std::cerr << "Failed to find header file '" << includeNamesData[i] << "'." << std::endl;
			exit( EXIT_FAILURE );
		}
		includeNameStrings[i] = includeNamesData[i].string();
		includeNames.push_back( includeNameStrings[i].c_str() );
		headers.push_back( headersData[i].c_str() );
	}

	if ( opts )
	{
		for ( const auto o : *opts )
			options.push_back( o );
	}

	options.push_back( "--use_fast_math" );
	std::string functionNameStorage = functionName;
	const char* functionNamePtr = functionNameStorage.c_str();

	std::vector<hiprtApiFunction> functions( 1 );
	hiprtError buildResult = hiprtBuildTraceKernels(
		ctxt,
		1,
		&functionNamePtr,
		sourceCode.c_str(),
		path.string().c_str(),
		static_cast<uint32_t>( headers.size() ),
		headers.data(),
		includeNames.data(),
		static_cast<uint32_t>( options.size() ),
		options.data(),
		numGeomTypes,
		numRayTypes,
		funcNameSets != nullptr ? funcNameSets->data() : nullptr,
		functions.data(),
		nullptr,
		false );

	if ( buildResult == hiprtSuccess )
	{
		functionOut = *reinterpret_cast<CUfunction*>( &functions[0] );
		return;
	}

	if ( funcNameSets != nullptr )
	{
		CHECK_HIPRT( buildResult );
	}

	const std::string compiler = findCudaCompiler();
	if ( compiler.empty() )
	{
		CHECK_HIPRT( buildResult );
	}

	const auto cacheKey = path.string();
	auto	   moduleIt = m_moduleCache.find( cacheKey );
	CUmodule   module	 = nullptr;
	if ( moduleIt == m_moduleCache.end() )
	{
		const auto sdkRoot = std::filesystem::path( HIPRTSDK_ROOT_DIR );
		const auto tempDir =
			std::filesystem::temp_directory_path() / ( "hiprtsdk-jit-" + std::to_string( std::hash<std::string>{}( cacheKey ) ) );
		std::filesystem::create_directories( tempDir );

		const auto cubinPath = tempDir / ( path.stem().string() + ".cubin" );
		std::ostringstream cmd;
		cmd << quoteShellArg( compiler ) << " -x cu " << quoteShellArg( path.string() ) << " -std=c++17 -O3 -cubin --use_fast_math"
			<< " -I" << quoteShellArg( sdkRoot.string() )
			<< " -I" << quoteShellArg( std::string( HIPRT_ROOT_DIRECTORY ) )
			<< " -I" << quoteShellArg( ( std::filesystem::path( HIPRT_ROOT_DIRECTORY ) / "contrib/Orochi" ).string() );
		for ( const auto option : options )
			cmd << " " << option;
		cmd << " -o " << quoteShellArg( cubinPath.string() );

		if ( std::system( cmd.str().c_str() ) != 0 )
		{
			CHECK_HIPRT( buildResult );
		}

		std::ifstream file( cubinPath, std::ios::in | std::ios::binary | std::ios::ate );
		if ( !file.is_open() )
		{
			CHECK_HIPRT( buildResult );
		}
		const size_t size = static_cast<size_t>( file.tellg() );
		file.seekg( 0, std::ios::beg );
		std::string cubin( size, '\0' );
		file.read( cubin.data(), static_cast<std::streamsize>( size ) );

		CHECK_ORO( cuModuleLoadData( &module, cubin.data() ) );
		m_moduleCache.emplace( cacheKey, module );
	}
	else
	{
		module = moduleIt->second;
	}

	CHECK_ORO( cuModuleGetFunction( &functionOut, module, functionName.c_str() ) );
}

void TutorialBase::launchKernel( CUfunction func, uint32_t nx, uint32_t ny, void** args )
{
	launchKernel( func, nx, ny, 8, 8, args );
}

void TutorialBase::launchKernel( CUfunction func, uint32_t nx, uint32_t ny, uint32_t bx, uint32_t by, void** args )
{
	hiprtInt3 nb;
	nb.x = ( nx + bx - 1 ) / bx;
	nb.y = ( ny + by - 1 ) / by;
	CHECK_ORO( cuLaunchKernel( func, nb.x, nb.y, 1, bx, by, 1, 0, nullptr, args, nullptr ) );
	CHECK_ORO( cuCtxSynchronize() );
}

void TutorialBase::writeImage( const std::string& path, uint32_t width, uint32_t height, uint8_t* pixels )
{
	std::vector<uint8_t> image( width * height * 4 );
	CHECK_ORO( cuMemcpyDtoH( image.data(), reinterpret_cast<CUdeviceptr>( pixels ), width * height * 4 ) );
	stbi_write_png( path.c_str(), width, height, 4, image.data(), width * 4 );
	std::cout << "image written at " << path.c_str() << std::endl;
}
