# OpenCL-Project

## About the project
This OpenCL program demonstrates the efficiency of the GPU and the CPU by measuring the execution time using profiling.
- IDE Used: ***Visual Studio 2019***

## ***IMPORTANT***
Since I only had 1 device, I used the same device to test my code but I did make sure that the code runs on CPU and GPU devices. To test I used line 127 instead of line 128.

## How to use the Program:
There are 3 macros defined for the program: ARRAY_SIZE, CPU_PARTITION, and GPU_PARTITION.
- ARRAY_SIZE is used to set the array size
- CPU_PARTITION and GPU_PARTITION are the array allocation in terms of percentage, this is used to partition the array 
  - example: 50 would be 50%, if array size is 1000 then it would be 500

## Kernel function approach:
Besides the required parameters I have a count, offset and num.
- count is the array size
- offset is used by the function to set boundaries
- num is used to identify if the cpu or the gpu is calling the function and according to this the partition for computation is used, the value of num is either 1 or 2.
	
The function starts by using num to see which partition should be used, if it is 1 then it checks if the current global id is less then count (array size) and offset. 

while on the other hand num 2 just adds offset to global id since partition 1 (CPU's part) ends there, then it checks if it is less than count (array size) then does the square root

## Output
