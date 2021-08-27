#!/usr/bin/env python3

import qarnot
import logging
from logging.config import fileConfig

def submit_task(param_dict):

    # Configure logger
    print("Configure logger")
    fileConfig("python_logging.conf")
    logger = logging.getLogger()
    logger.debug("Logger configured successfully")

    try:
        logger.debug("Connecting to Qarnot API...")
        conn = qarnot.Connection(client_token=param_dict['token'], cluster_url='https://api.qualif.qarnot.com/')

        logger.debug("Creating task...")
        task = conn.create_task('automl-mvp', 'auto-sklearn-cluster', int(param_dict['nodes']))

        # Create an input bucket and attach it to the task
        logger.debug("Provisionning input bucket...")
        input_bucket = conn.create_bucket('automl-mvp-input')
        input_bucket.sync_directory('input_binder/')
        task.resources.append(input_bucket)

        # Create a result bucket and attach it to the task
        output_bucket = conn.create_bucket('automl-mvp-output')
        task.results = output_bucket

        # Fill in task constants from the notebook form
        task.constants['DOCKER_TAG'] = "mvp"
        task.constants['TARGET_COL'] = param_dict['target']
        task.constants['POS_LABEL'] = param_dict['pos_label']
        task.constants['TOTAL_TIME_LIMIT'] = param_dict['total_time']
        task.constants['PER_RUN_TIME_LIMIT'] = param_dict['per_run_time']
        task.constants['N_CV_FOLDS'] = param_dict['cv']
        task.constants['ENSEMBLE_SIZE'] = param_dict['ensemble_size']
        task.constants['ENSEMBLE_NBEST'] = param_dict['ensemble_nbest']
        # Preprocess values in last tuples
        task.constants['INCLUDE_ESTIMATORS'] = " ".join(param_dict['incl_estim'])
        task.constants['EXCLUDE_ESTIMATORS'] = " ".join(param_dict['excl_estim'])
        task.constants['INCLUDE_PREPROCESSORS'] = " ".join(param_dict['incl_preproc'])
        task.constants['EXCLUDE_PREPROCESSORS'] = " ".join(param_dict['excl_preproc'])

        # take a snapshot every 5 seconds and ignore dask-worker-space directory
        task.snapshot(5)
        task.snapshot_blacklist='dask-worker-space'
        task.results_blacklist='dask-worker-space'

        # Submit the task
        logger.debug("Launching task...")    
        task.submit()

        # Wait for the task to be finished, and monitor the progress of its
        # deployment
        last_state = ''
        done = False
        while not done:
            if task.state != last_state:
                last_state = task.state
                print("** {}".format(last_state))

            # Wait for the task to complete, with a timeout of 5 seconds.
            # This will return True as soon as the task is complete, or False
            # after the timeout.
            done = task.wait(5)

        # Display errors on failure
        if task.state == 'Failure':
            print("** Errors: %s" % task.errors[0])
        
        # Download results in output folder
        task.download_results('output_binder')
    
    except Exception:
        logger.exception("An exception occured.")