import numpy as np
import os

# given a folder and object, generate MSE error for each path
if __name__ == '__main__':
    is_naive = False
    is_closed_loop = False
    object = "circle"
    even = True # should not be false
    foldername = f"data/{'naive' if is_naive else 'algo'}/{'closed_loop' if is_closed_loop else 'open_loop'}/{object}/{'even' if even else 'uneven'}/"
    if not os.path.exists(foldername):
        raise Exception(f"Folder {foldername} does not exist")
    
    thanos_translation_errors = []
    thanos_rotation_errors = []
    medusa_translation_errors = []
    medusa_rotation_errors = []
    print(len(os.listdir(foldername)))
    for filename in os.listdir(foldername):
        data = np.load(f"{foldername}/{filename}")
        qthanos = data[0]
        qmedusa = data[1]
        qgoal_thanos = data[2]
        qgoal_medusa = data[3]
        
        qthanos[2] *= 180/np.pi
        qgoal_thanos[2] *= 180/np.pi
        qmedusa[2] *= 180/np.pi
        qgoal_medusa[2] *= 180/np.pi
        
        # print(qmedusa[2], qgoal_medusa[2])
        print(qmedusa[:2], qgoal_medusa[:2])
        
        thanos_translation_error = np.linalg.norm(qthanos[:2] - qgoal_thanos[:2])
        thanos_rotation_error = np.abs(qthanos[2] - qgoal_thanos[2]) % 360
        if thanos_rotation_error > 180:
            thanos_rotation_error = 360 - thanos_rotation_error
            
        #wrap rotation error between 0 and np.pi
        
        medusa_translation_error = np.linalg.norm(qmedusa[:2] - qgoal_medusa[:2])
        
        medusa_rotation_error = np.abs(qmedusa[2] - qgoal_medusa[2]) % 360
        if medusa_rotation_error > 180:
            medusa_rotation_error = 360 - medusa_rotation_error
        
        
        thanos_translation_errors.append(thanos_translation_error)
        thanos_rotation_errors.append(thanos_rotation_error)
        medusa_translation_errors.append(medusa_translation_error)
        medusa_rotation_errors.append(medusa_rotation_error)
    thanos_translation_errors = np.array(thanos_translation_errors)
    thanos_rotation_errors = np.array(thanos_rotation_errors)
    medusa_translation_errors = np.array(medusa_translation_errors)
    medusa_rotation_errors = np.array(medusa_rotation_errors)
    
    # get ROOT MEAN SQUARED ERROR
    thanos_translation_rmse = np.sqrt(np.mean(thanos_translation_errors**2))
    thanos_rotation_rmse = np.sqrt(np.mean(thanos_rotation_errors**2))
    
    medusa_translation_rmse = np.sqrt(np.mean(medusa_translation_errors**2))
    medusa_rotation_rmse = np.sqrt(np.mean(medusa_rotation_errors**2))
    
    thanos_translation_stdev = np.std(thanos_translation_errors)
    thanos_rotation_stdev = np.std(thanos_rotation_errors)
    
    medusa_translation_stdev = np.std(medusa_translation_errors)
    medusa_rotation_stdev = np.std(medusa_rotation_errors)
    
    print(f"Results for {'closed loop' if is_closed_loop else 'open loop'}, {'naive' if is_naive else 'algo'}, {object}, {'even' if even else 'uneven'}")
    print(f"Thanos translation RMSE: {thanos_translation_rmse * 1000}")
    print(f"Thanos translation stdev: {thanos_translation_stdev * 1000}")
    print()
    print(f"Thanos rotation RMSE: {thanos_rotation_rmse}")
    print(f"Thanos rotation stdev: {thanos_rotation_stdev}")
    print()
    print(f"Medusa translation RMSE: {medusa_translation_rmse * 1000}")
    print(f"Medusa translation stdev: {medusa_translation_stdev * 1000}")
    print()
    print(f"Medusa rotation RMSE: {medusa_rotation_rmse}")
    print(f"Medusa rotation stdev: {medusa_rotation_stdev}")
    #get standard deviation
    pass