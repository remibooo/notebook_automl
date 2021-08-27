#!/usr/bin/env python3

import argparse
import logging
from logging.config import fileConfig
import qarnot
import sys
from datetime import datetime
from logging.config import fileConfig
#from requests import exceptions



def submit_task(token, repo, user, pwd, context_path, tag="latest"):

    # Configure logger
    print("Configure logger")
    fileConfig("python_logging.conf")
    logger = logging.getLogger()
    logger.debug("Logger configured successfully")

    try:
        logger.debug("Connecting to Qarnot API...")
        conn = qarnot.connection.Connection(client_token=token)

        # Create task, add a timestamp to differentiate different runs
        logger.debug("Creating task...")
        task = conn.create_task('{} kaniko build'.format(datetime.now()), 'docker-network', '1')

        # Sync input bucket
        # Context and dockerfile are both in context_path
        logger.debug("Provisionning input bucket...")
        logger.debug("Syncing {} content with the input bucket".format(context_path))
        input_bucket = conn.create_bucket('kaniko-input')
        input_bucket.sync_directory(context_path)
        task.resources.append(input_bucket)

        # Add the information to push the output image on the docker hub
        task.constants["REPO"] = repo
        task.constants["TAG"] = tag
        task.constants["USER"] = user
        task.constants["PWD"] = pwd

        # Add regular constants
        task.constants["DOCKER_REPO"] = "qarnotlab/kaniko"
        task.constants["DOCKER_TAG"] = "v1"
        task.constants['RESOURCES_PATH'] = '/kaniko/.docker'
        task.constants["DOCKER_CMD"] = "/bin/bash /opt/run_kaniko.sh ${REPO} ${TAG} ${USER} ${PWD}"

        # Submit the task
        logger.debug("Launching task...")
        task.submit()

        # Wait for the task to be finished, and monitor its progress
        last_state = ''
        done = False
        while not done:

            if task.state != last_state:
                last_state = task.state
                logger.debug("** {}".format(last_state))

            # Wait for the task to complete, with a timeout of 5 seconds.
            # This will return True as soon as the task is complete, or False
            # after the timeout.
            done = task.wait(5)

            # Display fresh stdout / stderr
            sys.stdout.write(task.fresh_stdout())
            sys.stderr.write(task.fresh_stderr())

        # Display errors on failure
        if task.state == 'Failure':
            logger.error("** Errors: %s" % task.errors[0])
        
    except Exception:
        logger.exception("An exception occured.")


# For debug
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--token", type=str,
                        help="Qarnot token", required=True)
    parser.add_argument("--repo", type=str,
                        help="Docker repo to push the built image", required=True)
    parser.add_argument("--user", type=str,
                        help="Docker user to push the built image", required=True)
    parser.add_argument("--pwd", type=str,
                        help="Docker password to push the built image", required=True)
    parser.add_argument("--context_path", type=str,
                        help="Local context ABSOLUTE path to add to bucket", required=True)
    parser.add_argument("--tag", type=str,
                        help="Docker tag to push the built image", required=False)
    args = parser.parse_args()

    # Download files
    try:
        submit_task(args.token, args.repo, args.user, args.pwd, args.context_path, args.tag)

    except Exception as e:
        print("Exception occured :", e)
