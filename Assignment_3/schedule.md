
Feb 18 - Mar 8  Assignment 3
---

- Feb 18-20: select 1~2 candidate machine learning methods for this task
- Feb 21-25: select the best parameters and features for each of the ML methods you are evaluating.
- Feb 26-27: Predict the likelihood to belong to class 1 for those instances provided in D2L (A3_test_dataset.tsv).
- Feb 28-Mar 3: finish the rest candidate machine learning method(can work together for this one)
- Feb Mar 4-5: generate the graphical representation of the cross-validation
- Feb Mar 6: do the testing
- Feb Mar 7: write the documentation
- Feb Mar 8: submit all the files***BEFORE DUE***

### The grading will be based on the performance of the models(AUPRC) so if you develop your own and I do my own and at the end we can compare and each one is better get submitted.  
If we need help from eachother in any part we can ask but I think doing it together is not a good idea.
Last time we spend a long time reading eachothers code. So here is my suggestion:  

We both do the it separetly and whenever both models are ready we can compare them and select the best. this is my schedule:  
 
    
#### Plan:
- Feb 19<br> 
	- Data preparation:
	- Selecting some features by correlation
	- Scale the features
- Feb 20 - 21
	*Apply KNN:
--- Prepare the AUPRC 
--- Apply KNN one and prepare AUPRC for that KNN
--- Repeat KNN for different neighbor numbers (a)
--- Return the best AUPRC and number of features for KNN (a)
--- Test the model on Test file
- Feb 22 
-- Find two other methods for classification- ( Probably Linear SVC and Bayes)
-- Use CV for each to find best parameter
-- Calculate AUPRC for each
-- Test the model on Test file
- Feb 22 
-- Test another model, like Ensemble or SVC
-- Test the model on Test file
-- Compare the result
- Feb 23
-- *Extra Day*
- Feb 24
-- Ask the prof any question
-- Finalize the model
- Feb 25
-- My Model is ready. Compare your model with mine whenever yours is ready
-- Apply CV on the model for reporting
- Feb 26 
-- Start the report
- Feb 27
-- Finish the report. Done!
 
