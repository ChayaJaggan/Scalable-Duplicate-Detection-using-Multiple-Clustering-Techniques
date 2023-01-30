# Scalable-Duplicate-Detection-using-Multiple-Clustering-Techniques
Code for scalable product duplicate detection using Locality Sensitive Hashing. All software is written in Pyhton 3.9 (https://www.python.org/). 
## Project Description 
For this project, the task was to create a scalable solution for product duplicate detection. Scalability means that the proposed algorithm could be implemented for multiple Web shops. Furthermore, the project required to use Locality Sensitive Hashing (LSH) to reduce the amount of comparisons. This code provides the implementation of LSH and a simplified version of MSMP and MSMP+. 
## Structure and Use of the Code
- data: 
  * TVs-all-merged.json: the data consisting of the product information for 1624 products. 
  * TVbrands.csv: a list of TV brands. 
 - Main.py:
 run this file to obtain the results of the implemented algorithm. 
 - Function.py:
 contains all the functions that are used in Main.py.  
 

## References
- Van Bezu, R., Borst, S., Rijkse, R., Verhagen, J., Vandic, D., Frasincar, F.: Multi-
component similarity method for web product duplicate detection. In: 30th ACM
Symposium on Applied Computing (SAC 2015). pp. 761–768. ACM (2015)
- Van Dam, I., van Ginkel, G., Kuipers, W., Nijenhuis, N., Vandic, D., Frasincar, F.:
Duplicate detection in web shops using LSH to reduce the number of computations.
In: 31th ACM Symposium on Applied Computing (SAC 2016). pp. 772–779. ACM (2016)
- Hartveld, A., Keulen, M.v., Mathol, D., Noort, T.v., Plaatsman, T., Frasincar,
F., Schouten, K.: An LSH-based model-words-driven product duplicate detection
method. In: 30th International Conference on Advanced Information Systems Engi-
neering (CAiSE 2018). Lecture Notes in Computer Science, vol. 10816, pp. 149–161.
Springer (2018)
- Indyk, P., Motwani, R.: Approximate nearest neighbors: towards removing the
curse of dimensionality. In: Thirtieth Annual ACM Symposium on Theory of Com-
puting (STOC 1998). pp. 604–613. ACM (1998)
