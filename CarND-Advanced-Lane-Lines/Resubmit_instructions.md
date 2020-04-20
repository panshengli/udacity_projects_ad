### Resubmission instructions
Hello reviewers, 
   Thank you for pointing out the shortcomings and improvement suggestions of my project. For the first review, I have optimized the following two points.
   Firstly, I optimized the pixel position of projection transformation, that is the region of interest, it is really hard to adjustment and spend servals hours to fitting. More details shown in the `project_handbook.md`.
   Secondly, In combination with the projection transformation parameters, I adjusted the margin of the lane line.If the margin is too large, the curvature error of the lane line will increase, resulting lane is lost; if the margin is too small, resulting in the inability to track the lane line of the previous frame, and the final lane line matching error. In this case, i set margin of lane is 55.
   The final results show that the lane line detection results have a good improvement compared with the first submission, and there is no staggered lane line. However, in some area where the lane line is not clear, the lane line detection will become wider, and the overall results are relatively stable.
   Thank you for your time.
