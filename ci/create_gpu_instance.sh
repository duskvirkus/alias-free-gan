#!/bin/bash

c_command="gcloud compute instances create ci-gpu-instance --source-instance-template ci-gpu-template --zone us-central1-a";
$c_command;
stat=$?;

if [ $stat -eq 0 ];then
    export INSTANCE_ZONE=us-central1-a
    echo "created on zone us-central1-a"
    exit 0
else
    echo "zone us-central1-a failed"
fi

c_command="gcloud compute instances create ci-gpu-instance --source-instance-template ci-gpu-template --zone us-central1-b";
$c_command;
stat=$?;

if [ $stat -eq 0 ];then
    export INSTANCE_ZONE=us-central1-b
    echo "created on zone us-central1-b"
    exit 0
else
    echo "zone us-central1-b failed"
fi

c_command="gcloud compute instances create ci-gpu-instance --source-instance-template ci-gpu-template --zone us-central1-c";
$c_command;
stat=$?;

if [ $stat -eq 0 ];then
    export INSTANCE_ZONE=us-central1-c
    echo "created on zone us-central1-c"
    exit 0
else
    echo "zone us-central1-c failed"
fi

c_command="gcloud compute instances create ci-gpu-instance --source-instance-template ci-gpu-template --zone us-central1-f";
$c_command;
stat=$?;

if [ $stat -eq 0 ];then
    export INSTANCE_ZONE=us-central1-f
    echo "created on zone us-central1-f"
    exit 0
else
    echo "zone us-central1-f failed"
fi

echo "no zones worked!"
exit 1

# gcloud compute instances create ci-gpu-instance --source-instance-template ci-gpu-template --zone us-central1-f

