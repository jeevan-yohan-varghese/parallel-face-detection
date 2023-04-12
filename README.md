# Parallel Face Detection

Proposing a parallelized video analysis system for face identification. The system is based on a state-of-the-art face identification algorithm and is designed to leverage both CPU and GPU resources to achieve high throughput. By distributing the workload across multiple computing nodes, the system is scalable and can achieve linear speedup with the number of nodes. The proposed system can be applied to a wide range of real-world scenarios, such as video surveillance applications. The report will describe the system architecture, algorithms used for face identification, and parallelization techniques employed. Experimental results will be presented to demonstrate the performance improvements achieved by the parallelized system compared to a single-node implementation. The project aims to contribute to the development of more efficient and effective video analysis systems in various fields.

The code will identify faces in a parallel manner and export the identified faces as jpg files

<b>Dependencies</b>
- OpenCV
- OpenMP
