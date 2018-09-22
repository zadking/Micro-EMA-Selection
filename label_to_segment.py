import numpy as np
import pandas as pd


#Assumption column names returned from segment_bySlidingWindow
def ratio_by_segment(inputDfRow, annotations, threshold):    
    '''
    Finds the label of the row that includes segment start time as index and end time in the column "End_time". 
    If segment is not found in one of the activity durations, it will have None value. If segment covers end of one activity 
    and beginning of following activity, its label is determined by threshold value. Whichever activity covers more than threshold
    percentage of the segment, segment is labelled as that activity.
    
    Parameters
    ----------
    inputDfRow:       series that contain start(index) and end time of one segment
    annotations:      dataFrame that has start(index) and end times of labels/activities 
    threshold:        int, between 0 and 1, minimum segment length needed when segment is between two activities
    
    Return
    ------
    activityLabel:    string or None, if found returns name of the activity, else (if segment is not in any activity) returns
                      None.

    '''       
    startTime = inputDfRow.name
    endTime = inputDfRow.End_time    

    
    nextLabel = False
    for activityStart, row in annotations.iterrows():
        
        activityEnd = row.values[0]
        activityLabel = row.values[1]
        
        #Second subsegment size is >= threshold
        if nextLabel: 
            return activityLabel        

        if startTime >= activityStart and  startTime < activityEnd:
            
            #Segment is entirely inside one activity
            if endTime <= activityEnd: 
                return activityLabel
            
            #Segment is only partly inside activity, need to check how much is in and compare to threshold
            else:
                length = endTime - startTime
                subSegment1 = activityEnd - startTime            
                
                portion = float(subSegment1) / length

                if portion >= threshold:
                    return activityLabel
                else:
                    nextLabel = True
        
        #Segment belongs to a time before any activity
        elif startTime < activityStart and endTime < activityStart: 
            return None
        
        #Segment starts when there is no activity but ends inside an activity
        elif startTime < activityStart and endTime < activityEnd:
            length = endTime - startTime
            subSegment1 = activityStart - startTime

            portion = float(subSegment1) / length

            if portion >= threshold:
                return None
            else:
                return activityLabel


def label_to_segment(inputDf, annotationsDf, threshold=0.5, inplace=False):
    '''
    Goes over each segment in inputDf and finds label of each segment from the annotations dataframe 
    
    Parameters
    ----------
    inputDf:         dataFrame that is output of segment_by_sliding_window
    annotationsDf:   dataFrame that has start(index) and end times of labels/activities    
    threshold:       int, between 0 and 1, minimum segment length needed when segment is between two activities (0.5 by default)
    inplace:         boolean, if True inputDf will be modified inplace, else new dataFrame (outputDf) will be returned (False by default)

    Return
    ------
    Depends on the input parameter "inplace".

    '''    

    if threshold <= 0 or threshold > 1:
        print "ERROR: Threshold cannot be <= 0 or >1. Calculating with default threshold (0.5)."
        threshold = 0.5

    labels = []

       
    labels = inputDf.apply(ratio_by_segment, args=(annotationsDf, threshold), axis=1)

    if inplace:
        inputDf["Label"] = labels
        return None
    else:
        outputDf = inputDf
        outputDf["Label"] = labels
        return outputDf

if __name__ == "__main__":
	anno = "annotations.csv"
	annotationData = pd.read_csv(anno)

	#PREPROCESS OF FILES IS NEEDED
	start = annotationData['Start Timestamp (ms)']
	end = annotationData['Stop Timestamp (ms)'].values
	act = annotationData['EventType'].values    
	annotationsMinimized = pd.DataFrame(index = start, data = end, columns=["Stop Timestamp (ms)"])
	annotationsMinimized['Activity'] = act

	segments_slidingwindow = pd.DataFrame(data = [4,6,8,10,12,14,16,18,20,22,24], index=[0,2,4,6,8,10,12,14,16,18,20], columns=['End_time'])

	returnDf = label_to_segment(segments_slidingwindow, annotationsMinimized)
	print returnDf