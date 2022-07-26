// Add you host code
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <math.h>

#define ARRAY_SIZE 1000
#define CPU_PARTITION 50
#define GPU_PARTITION 50

void callStatus(char* task, int err)
{
	printf("%s: [%s] (%d)\n", task, err == 0 ? "OK" : "FAILED", err);
}

char* getOpenCLProgramFromFile(const char* filename) {
	FILE* programFile;
	char* programSource;
	size_t program_source_size;

	programFile = fopen(filename, "rb");
	if (!programFile)
	{
		printf("Failed to load kernel \n");
		exit(1);
	}

	// Get file length and allocate space to read the contents
	fseek(programFile, 0, SEEK_END);
	program_source_size = ftell(programFile);
	programSource = (char*)malloc(program_source_size + 1);

	// Rewind the file to the beginning, read the contents, and add a null terminator
	rewind(programFile);
	fread(programSource, sizeof(char), program_source_size, programFile);
	programSource[program_source_size] = '\0';

	fclose(programFile);
	return programSource;
}

cl_device_id getFirstDeviceByType(cl_device_type device_type)
{
	cl_uint			numPlatforms;
	cl_uint			numDevice;

	cl_platform_id* platforms;
	cl_device_id*	devices;

	// Get the number of platforms
	clGetPlatformIDs(0, NULL, &numPlatforms);
	platforms = (cl_platform_id*)malloc(numPlatforms * sizeof(cl_platform_id));
	clGetPlatformIDs(numPlatforms, platforms, NULL);

	// Loop through each platform 
	for (int i = 0; i < numPlatforms; i++)
	{
		// Get the count of each device of the platform of the type passed as a parameter
		clGetDeviceIDs(platforms[i], device_type, 0, NULL, &numDevice);
		devices = (cl_device_id*)malloc(sizeof(cl_device_id) * numDevice);
		clGetDeviceIDs(platforms[i], device_type, numDevice, devices, NULL);
		
		// returns first device if platform has atleast 1 device
		if (numDevice > 0)
			return devices[0];

		free(devices);
	}
	
	free(platforms);
	return NULL;
}

void printDeviceInfo(cl_device_id device)
{
	size_t			textSize;
	char*			text;
	cl_device_type	temp;
	cl_uint			maxComputeUnits;

	printf("-------- Device Info --------\n");

	// Get device name
	clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &textSize);
	text = (char*)malloc(textSize);
	clGetDeviceInfo(device, CL_DEVICE_NAME, textSize, text, NULL);
	printf("Name: %s\n", text);
	free(text);

	// Get device vendor
	clGetDeviceInfo(device, CL_DEVICE_VENDOR, 0, NULL, &textSize);
	text = (char*)malloc(textSize);
	clGetDeviceInfo(device, CL_DEVICE_VENDOR, textSize, text, NULL);
	printf("Vendor: %s\n", text);
	free(text);
	
	// Get device type
	clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(temp), &temp, NULL);
	if (temp == CL_DEVICE_TYPE_GPU)
		printf("Type: GPU\n");
	else if (temp == CL_DEVICE_TYPE_CPU)
		printf("Type: CPU\n");

	// Get device compute units
	clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(maxComputeUnits), &maxComputeUnits, NULL);
	printf("Cores: %d\n\n", maxComputeUnits);

}

int main()
{
	cl_int			err;
	cl_device_id	deviceGPU;
	cl_device_id	deviceCPU;

	int				offset;
	int*			input;
	float*			output, *output2;
	const char*		initArrayKernel;


	// The function goes through the available platforms and devices
	// and returns the first device according to the parameter (CPU & GPU)
	deviceGPU = getFirstDeviceByType(CL_DEVICE_TYPE_GPU);
	// ******** Since I only had 1 device I did it this way to check my code**********
	//deviceCPU = getFirstDeviceByType(CL_DEVICE_TYPE_GPU);
	deviceCPU = getFirstDeviceByType(CL_DEVICE_TYPE_CPU);

	// A check to see if a device has been found for both CPU & GPU
	if (deviceGPU == NULL || deviceCPU == NULL)
	{
		if (deviceCPU == NULL)
			printf("CPU device not found,");
		if (deviceGPU == NULL)
			printf("GPU device not found,");
		printf(" the program requires 1 GPU and 1 CPU\n");
		goto end_of_program;
	}

	// identifing the device acquired from the above function
	printDeviceInfo(deviceGPU);
	printDeviceInfo(deviceCPU);

	// Allocate memory and Initialize input data 
	// with randomizer w/ seed 12345
	input = (int *)malloc(sizeof(int) * ARRAY_SIZE);
	output = (float *)malloc(sizeof(float) * ARRAY_SIZE);
	output2 = (float *)malloc(sizeof(float) * ARRAY_SIZE);
	offset = (float)ARRAY_SIZE * ((float)CPU_PARTITION / 100);
	printf("Offset set to: %d = %d * ( %d / 100)\n", offset, ARRAY_SIZE, CPU_PARTITION);

	srand(12345);
	for (int i = 0; i < ARRAY_SIZE; i++)
		input[i] = rand() % 1000;

	// Seting up the OpenCL framework required for each of the device
	cl_context   		contextGPU;
	cl_context   		contextCPU;

	cl_program			programGPU;
	cl_kernel			kernelGPU;
	cl_command_queue	commandQueueGPU;

	cl_program			programCPU;
	cl_kernel			kernelCPU;
	cl_command_queue	commandQueueCPU;

	cl_mem       		bufferGPU_IN;
	cl_mem       		bufferGPU_OUT;

	cl_mem       		bufferCPU_IN;
	cl_mem       		bufferCPU_OUT;

	printf("-------- Set up --------\n");

	// Get program from file
	initArrayKernel = getOpenCLProgramFromFile("KernelFile.txt");

	// create a context
	//CPU
	contextCPU = clCreateContext(NULL, 1, &deviceCPU, NULL, NULL, &err);
	callStatus("Creating a CPU Context", err);
	//GPU
	contextGPU = clCreateContext(NULL, 1, &deviceGPU, NULL, NULL, &err);
	callStatus("Creating a GPU Context", err);
	


	// create a queue
	//CPU
	commandQueueCPU = clCreateCommandQueue(contextCPU, deviceCPU, CL_QUEUE_PROFILING_ENABLE, &err);
	callStatus("Creating a CPU Command Queue", err);
	//GPU
	commandQueueGPU = clCreateCommandQueue(contextGPU, deviceGPU, CL_QUEUE_PROFILING_ENABLE, &err);
	callStatus("Creating a GPU Command Queue", err);
	

	// create program from source
	//CPU
	programCPU = clCreateProgramWithSource(contextCPU, 1, &initArrayKernel, NULL, &err);
	callStatus("Creating CPU Program from Source", err);
	//GPU
	programGPU = clCreateProgramWithSource(contextGPU, 1, &initArrayKernel, NULL, &err);
	callStatus("Creating GPU Program from Source", err);
	
	// build the program
	err = clBuildProgram(programCPU, 0, NULL, NULL, NULL, NULL);
	callStatus("Building CPU Program", err);
	err = clBuildProgram(programGPU, 0, NULL, NULL, NULL, NULL);
	callStatus("Building GPU Program", err);

	// fetch a kernel
	//CPU
	kernelCPU = clCreateKernel(programCPU, "squareRoot", &err);
	callStatus("Creating Kernel from CPU Program", err);
	//GPU
	kernelGPU = clCreateKernel(programGPU, "squareRoot", &err);
	callStatus("Creating Kernel from GPU Program", err);

	// create device memory for the array
	//CPU
	bufferCPU_IN = clCreateBuffer(contextCPU, CL_MEM_READ_WRITE, sizeof(int) * ARRAY_SIZE, NULL, &err);
	callStatus("Creating CPU IN Buffer", err);
	bufferCPU_OUT = clCreateBuffer(contextCPU, CL_MEM_READ_WRITE, sizeof(float) * ARRAY_SIZE, NULL, &err);
	callStatus("Creating CPU OUT Buffer", err);
	//GPU
	bufferGPU_IN = clCreateBuffer(contextGPU, CL_MEM_READ_WRITE, sizeof(int) * ARRAY_SIZE, NULL, &err);
	callStatus("Creating GPU IN Buffer", err);
	bufferGPU_OUT = clCreateBuffer(contextGPU, CL_MEM_READ_WRITE, sizeof(float) * ARRAY_SIZE, NULL, &err);
	callStatus("Creating GPU OUT Buffer", err);

	// transfer host data to device memory for GPU then CPU
	// CPU
	err = clEnqueueWriteBuffer(commandQueueCPU, bufferCPU_IN, CL_TRUE, 0, sizeof(int) * ARRAY_SIZE, input, 0, NULL, NULL);
	callStatus("Write to CPU Buffer IN", err);
	err = clEnqueueWriteBuffer(commandQueueCPU, bufferCPU_OUT, CL_TRUE, 0, sizeof(float) * ARRAY_SIZE, output, 0, NULL, NULL);
	callStatus("Write to CPU Buffer OUT", err);
	//GPU
	err = clEnqueueWriteBuffer(commandQueueGPU, bufferGPU_IN, CL_TRUE, 0, sizeof(int) * ARRAY_SIZE, input, 0, NULL, NULL);
	callStatus("Write to GPU Buffer IN", err);
	err = clEnqueueWriteBuffer(commandQueueGPU, bufferGPU_OUT, CL_TRUE, 0, sizeof(float) * ARRAY_SIZE, output, 0, NULL, NULL);
	callStatus("Write to GPU Buffer OUT", err);

	// set kernel's arguments
	// __global int* inputBuffer, __global float* outputBuffer, const unsigned int offset, const unsigned int count, const unsigned int num
	unsigned int count = ARRAY_SIZE;
	unsigned int type1 = 1;
	unsigned int type2 = 2;
// Setting CPU kernel arguments
	err = clSetKernelArg(kernelCPU, 0, sizeof(cl_mem), &bufferCPU_IN);
	callStatus("CPU - Setting Parameter 0 (__global int* inputBuffer)", err);
	err = clSetKernelArg(kernelCPU, 1, sizeof(cl_mem), &bufferCPU_OUT);
	callStatus("CPU - Setting Parameter 1 (__global float* outputBuffer)", err);
	err = clSetKernelArg(kernelCPU, 2, sizeof(unsigned int), &offset);
	callStatus("CPU - Setting Parameter 2 (const unsigned int offset)", err);
	err = clSetKernelArg(kernelCPU, 3, sizeof(unsigned int), &count);
	callStatus("CPU - Setting Parameter 3 (const unsigned int count)", err);
	err = clSetKernelArg(kernelCPU, 4, sizeof(unsigned int), &type1);
	callStatus("CPU - Setting Parameter 4 (const unsigned int num)", err);

// Setting GPU kernel arguments
	err = clSetKernelArg(kernelGPU, 0, sizeof(cl_mem), &bufferGPU_IN);
	callStatus("GPU - Setting Parameter 0 (__global int* inputBuffer)", err);
	err = clSetKernelArg(kernelGPU, 1, sizeof(cl_mem), &bufferGPU_OUT);
	callStatus("GPU - Setting Parameter 1 (__global float* outputBuffer)", err);
	err = clSetKernelArg(kernelGPU, 2, sizeof(unsigned int), &offset);
	callStatus("GPU - Setting Parameter 2 (const unsigned int offset)", err);
	err = clSetKernelArg(kernelGPU, 3, sizeof(unsigned int), &count);
	callStatus("GPU - Setting Parameter 3 (const unsigned int count)", err);
	err = clSetKernelArg(kernelGPU, 4, sizeof(unsigned int), &type2);
	callStatus("GPU - Setting Parameter 4 (const unsigned int num)", err);

	size_t local = 10;
	size_t global = ceil(count / (float)local) * local;

	// enqueue kernel runs kernel with all work items
	printf("\n-------- %d%% CPU - %d%% GPU --------\n", CPU_PARTITION, GPU_PARTITION);
	cl_event kernel_run_event_GPU, kernel_run_event_CPU;
	err = clEnqueueNDRangeKernel(commandQueueGPU, kernelGPU, 1, NULL, &global, &local, 0, NULL, &kernel_run_event_GPU);
	callStatus("Start GPU Kernel", err);
	err = clEnqueueNDRangeKernel(commandQueueCPU, kernelCPU, 1, NULL, &global, &local, 0, NULL, &kernel_run_event_CPU);
	callStatus("Start CPU Kernel", err);

	// Blocks until all previously queued OpenCL commands finished
	err = clFinish(commandQueueGPU);
	callStatus("\nWaiting for GPU Work to Finish", err);
	err = clFinish(commandQueueCPU);
	callStatus("Waiting for CPU Work to Finish", err);

	// Calculating Time for GPU and CPU
	cl_ulong startGPU, endGPU, startCPU, endCPU;

	clGetEventProfilingInfo(kernel_run_event_GPU, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &startGPU, NULL);
	clGetEventProfilingInfo(kernel_run_event_GPU, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endGPU, NULL);

	clGetEventProfilingInfo(kernel_run_event_CPU, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &startCPU, NULL);
	clGetEventProfilingInfo(kernel_run_event_CPU, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endCPU, NULL);

	printf("\n-------- Timing --------\n", CPU_PARTITION, GPU_PARTITION);
	printf("CPU Time Taken: %f ms\n", (endCPU - startCPU) / 1000000.0);
	printf("GPU Time Taken: %f ms\n", (endGPU - startGPU) / 1000000.0);



	// read output from device memory to host memory
	clEnqueueReadBuffer(commandQueueCPU, bufferCPU_OUT, CL_TRUE, 0, sizeof(float)* ARRAY_SIZE, output, 0, NULL, NULL);
	clEnqueueReadBuffer(commandQueueGPU, bufferGPU_OUT, CL_TRUE, 0, sizeof(float)* ARRAY_SIZE, output2, 0, NULL, NULL);

	// Check CPU and GPU output
	int flag = 0;
	for (int i = 0; i < offset; i++)
	{
		// Round the values to 2 decimal places then check if they are same
		// if they arent then raise a flag which will show that the output isnt correct
		float value1 = (int)(output[i] * 100 + .5);
		value1 /= 100;
		float value2 = (int)(sqrt(input[i]) * 100 + .5);
		value2 /= 100;

		if (value1 != value2)
		{
			printf("CPU Output wrong\n");
			flag = 1;
			break;
		}
		
	}

	for (int i = offset; i < count; i++)
	{
		// Round the values to 2 decimal places then check if they are same
		// if they arent then raise a flag which will show that the output isnt correct
		float value1 = (int)(output2[i] * 100 + .5);
		value1 /= 100;
		float value2 = (int)(sqrt(input[i]) * 100 + .5);
		value2 /= 100;
		
		if (value1 != value2)
		{
			printf("GPU Output wrong\n");
			flag = 1;
			break;
		}
		
	}

	// If no flags raised then outputs are correct
	if (flag == 0)
		printf("\nGPU and CPU output correct\n");


	// release OpenCL objects
	clReleaseMemObject(bufferGPU_IN);
	clReleaseMemObject(bufferGPU_OUT);
	clReleaseMemObject(bufferCPU_IN);
	clReleaseMemObject(bufferCPU_OUT);
	clReleaseEvent(kernel_run_event_CPU);
	clReleaseEvent(kernel_run_event_GPU);
	clReleaseProgram(programGPU);
	clReleaseProgram(programCPU);
	clReleaseKernel(kernelGPU);
	clReleaseKernel(kernelCPU);
	clReleaseCommandQueue(commandQueueGPU);
	clReleaseCommandQueue(commandQueueCPU);
	clReleaseContext(contextGPU);
	clReleaseContext(contextCPU);

	// free allocated memory
	free(input);
	free(output);
	free(output2);

end_of_program:
	system("PAUSE");
	return(0);
}