import boto3
import os
import sagemaker
import sagemaker.session
import datetime
import json
import pathlib
import yaml
from sagemaker.xgboost import XGBoostProcessor
from sagemaker.estimator import Estimator
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.parameters import (
    ParameterString,
)
from sagemaker.processing import ScriptProcessor
from sagemaker.workflow.properties import PropertyFile
from sagemaker.estimator import Estimator
from sagemaker.model import Model
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.functions import JsonGet, Join
from sagemaker.workflow.condition_step import (
    ConditionStep,
    JsonGet
)

os.chdir(pathlib.Path(__file__).parent.resolve())

region = boto3.Session().region_name
sagemaker_session = sagemaker.session.Session()
pipeline_session = PipelineSession()

def get_data_path(env, de_config_bucket, de_config_key):
    s3_client = boto3.client('s3')
    bucket = f"{de_config_bucket}-{env}"
    response = s3_client.get_object(Bucket=bucket, Key=de_config_key)
    configfile = yaml.safe_load(response["Body"])
    return configfile['common_reference_data'][''].replace('${ENV}', env)

def handler(env):

    indications = ["pvr"]

    pipeline_steps={}

    for indication in indications:
        try:
            data = open(f"config/config_{indication}.json")
            param_json = json.load(data)
            data.close()
        except:
            data = open(f"parameters/config_{indication}.json")
            param_json = json.load(data)
            data.close()
        
        acc_no = boto3.client('sts').get_caller_identity().get('Account')
        role = f"arn:aws:iam::{acc_no}:role/-sagemaker-role"
        print(role)

        
        project = param_json['training_handler_config']['project']
        model_package_group_name = param_json['training_handler_config']['model_package_group_name_prod']
        curr_time = datetime.datetime.today().strftime('%Y-%m-%d')
        train_pipeline_name = param_json['training_handler_config']['train_pipeline_name']
        # train_pipeline_name_main = param_json['training_handler_config']['train_pipeline_name']
        register_step_name = param_json['training_handler_config']['register_step_name']
        train_step_name = param_json['training_handler_config']['train_step_name']
        # pipeline_name = param_json['training_handler_config']['pipeline_name']
        preprocess_step_name = param_json['training_handler_config']['preprocess_step_name']
        params_dir = param_json['training_handler_config']['input_params']
        input_training_config_path = param_json['training_handler_config']['input_training_config_path']
        input_config_path = param_json['training_handler_config']['input_config_path']
        # input_sample_output_path = param_json['training_handler_config']['input_sample_output_path']
        # training_output_path = param_json['training_handler_config']['training_output_path']

        model_status_default = param_json['training_handler_config']['model_status_default']

        #setting the environment specific param variables
        write_bucket = param_json['training_handler_config']['write_bucket'].replace("-#env",f"-{env}")
        model_path = param_json['training_handler_config']['model_path'].replace("-#env",f"-{env}")
        eval_output = param_json['training_handler_config']['eval_output'].replace("-#env",f"-{env}")

        #setting environment variables to script level parameters
        try:
            data = open(f"config/config_{project}.json")
            script_params = json.load(data)
            data.close()
        except:
            data = open(f"parameters/config_{project}.json")
            script_params = json.load(data)
            data.close()

        script_params['training_components_config']['S3_BUCKET'] = script_params['training_components_config']['S3_BUCKET'].replace("-#env",f"-{env}")
        script_params['training_components_config']['default_bucket'] = script_params['training_components_config']['default_bucket'].replace("-#env",f"-{env}")
        script_params['training_components_config']['preprocess_path'] = script_params['training_components_config']['preprocess_path'].replace("-#env",f"-{env}")
        script_params['training_components_config']['sampled_path'] = script_params['training_components_config']['sampled_path'].replace("-#env",f"-{env}")
        de_config_bucket = param_json["training_components_config"]["de_config_bucket"]
        de_config_key = param_json["training_components_config"]["de_config_key"]
        datalake_bucket = param_json["training_handler_config"]["prod_datalake_bucket"]
        preprocess_path = script_params['training_components_config']['preprocess_path']
        sampled_path = script_params['training_components_config']['sampled_path']
        oncemr_data = get_data_path(env, de_config_bucket, de_config_key)
        script_params['training_components_config']['integrated_read_path'] = oncemr_data
        
        s3 = boto3.resource('s3')
        s3object = s3.Object(write_bucket, f"{params_dir}config_{project}.json" )
        s3object.put(Body=(bytes(json.dumps(script_params).encode('UTF-8'))))
        
        # saving the date for the latest train run
        date_path = '/'.join(params_dir.split('/')[:-3])
        date_dict = {"date" : curr_time}
        s3object_date = s3.Object(datalake_bucket, f"{date_path}/date_{project}.json" )
        s3object_date.put(Body=(bytes(json.dumps(date_dict).encode('UTF-8'))))
        print("Environment specific parameters have been written")
        
        # PIPELINE ---------------------------------------------------------------------------------------
        print("=====================>")
        print("pipeline creation started")
        #Initialize the XGBoostProcessor


        xgb = XGBoostProcessor(
            framework_version='1.7-1',
            py_version="py3",
            role=role,
            instance_type='ml.m5.24xlarge',
            instance_count=1,
            sagemaker_session=PipelineSession(),
            base_job_name='PIE-Preprocess-XGB'
        )
        
        step_args_pre = xgb.run(
            code = param_json['training_handler_config']['preprocess'],
            source_dir = 'preprocess',
            inputs=[
                ProcessingInput(
                    input_name='params',
                    source=f's3://{write_bucket}/{params_dir}',
                    destination='/opt/ml/processing/input/parameters'
                ),
                ProcessingInput(
                    input_name='oncemr',
                    source=f'{oncemr_data}',
                    destination='/opt/ml/processing/input/oncemr/oncemr.parquet'
                )
            ],
            outputs=[
                ProcessingOutput(
                    output_name='preprocess_output_s3',
                    source='/opt/ml/processing/output/preprocess/',
                    destination= f'{preprocess_path}/{curr_time}/'
                )
            ],
            arguments=[
                "--INDICATION",indication,
                "--INPUT_CONFIG_PATH",input_config_path,
                "--ENV", env,
            ]
        )
        pipeline_steps[f'preprocessing_{indication}'] = ProcessingStep(
                                                                        name = preprocess_step_name+"-" + indication,
                                                                        step_args = step_args_pre
                                                                    )
        

        step_args_sampling = xgb.run(
            code = param_json['training_handler_config']['sampling'],
            source_dir = 'sampling',
            inputs=[
                    ProcessingInput(
                        input_name='params',
                        source=f's3://{write_bucket}/{params_dir}',
                        destination='/opt/ml/processing/input/parameters'
                    ),
                    ProcessingInput(
                        input_name='preprocess_output_s3',
                        source=f'{preprocess_path}/{curr_time}/',
                        destination='/opt/ml/processing/input/preprocessed'
                    )
            ],
            outputs=[
                    ProcessingOutput(
                        output_name='sampling_output_s3',
                        source='/opt/ml/processing/output/sampling/',
                        destination= f'{sampled_path}/{curr_time}/'
                    )
            ],
            arguments=[
                    "--INDICATION",indication,
                    "--INPUT_CONFIG_PATH",input_config_path,
                    "--ENV", env,
            ],
        )

        pipeline_steps[f'sampling_{indication}'] = ProcessingStep(
            name= param_json['training_handler_config']['sampling_step_name'] +"-" +indication,
            step_args=step_args_sampling
        )
        
        pipeline_steps[f'sampling_{indication}'].add_depends_on([pipeline_steps[f'preprocessing_{indication}']])

        print("=====================>")
        print("Component 1 created : Integrated data processing")

        # Defining Component 2 : Training ------------------------------------------------------------------
        image_uri = sagemaker.image_uris.retrieve(
                        framework = "xgboost",
                        region = region,
                        version = "1.7-1",
                        py_version = "py3",
                        instance_type = "ml.m5.24xlarge"
                    )
        xgb_train = Estimator(
                        image_uri = image_uri,
                        instance_type = "ml.m5.24xlarge",
                        instance_count = 1,
                        output_path = model_path,
                        role = role,
                        source_dir = 'train',
                        entry_point = param_json['training_handler_config']['trainpy'],
                        hyperparameters = {"indication": indication,
                            "input_training_config_path": input_training_config_path,
                            "curr_time":curr_time,
                            "env":env
                            },
                    )
        pipeline_steps[f'train_{indication}'] = TrainingStep(
                                    name = train_step_name + "-"+indication,
                                    estimator = xgb_train,
                                    inputs = {
                                        "sampled_data": TrainingInput(
                                            s3_data= f'{sampled_path}/{curr_time}/'
                                            
                                        ),
                                        "parameters" : TrainingInput(
                                            s3_data = f's3://{write_bucket}/{params_dir}'
                                        )
                                    }
                                )
        pipeline_steps[f'train_{indication}'].add_depends_on([pipeline_steps[f'sampling_{indication}']])

        #pipeline_steps[f'evaluate_{indication}'].append(step_evaluate_model)

        # Defining Register Model Step------------------------------------------------------------------
        
        metadata_properties = dict(prod='PendingManualApproval', uat='PendingManualApproval', test='PendingManualApproval',dev='PendingManualApproval')

        model_approval_status = ParameterString(
                                                    name="ModelApprovalStatus",
                                                    default_value=model_status_default
                                                )

        model = Model(
                        image_uri=xgb_train.training_image_uri(),
                        model_data=pipeline_steps[f'train_{indication}'].properties.ModelArtifacts.S3ModelArtifacts,
                        sagemaker_session=PipelineSession(),
                        role=role,
                        model_kms_key = ,
                    )
        pipeline_steps[f'register_{indication}'] = ModelStep(
                                    name = register_step_name+"-" +indication,
                                    step_args=model.register(
                                        content_types=["text/csv"],
                                        response_types=["text/csv"],
                                        model_package_group_name = model_package_group_name,
                                        approval_status=model_approval_status,
                                        #model_metrics=pipeline_model_metrics,
                                        customer_metadata_properties = metadata_properties
                                    )
                                )  
        steps=[]
                ## Condtion step

        # cond_gte = ConditionGreaterThanOrEqualTo(
        #     left=JsonGet(
        #         step=pipeline_steps[f'evaluate_{indication}'],
        #         property_file=evaluation_report,
        #         json_path="register_flag.eligible.value"
        #     ),
        #     right= 1
        # )
        # pipeline_steps[f'step_condition_{indication}'] = ConditionStep(
        #     name="pie-bladder-cond",
        #     conditions=[cond_gte],
        #     if_steps=[pipeline_steps[f'register_{indication}']],
        #     else_steps=[]
        # )

        # pipeline_steps.append(step_cond)
        #pipeline_steps[f'step_condition_{indication}'].append(step_cond)

        steps.append(pipeline_steps[f'preprocessing_{indication}'])
        steps.append(pipeline_steps[f'sampling_{indication}'])
        steps.append(pipeline_steps[f'train_{indication}'])
        # steps.append(pipeline_steps[f'evaluate_{indication}'])
        # steps.append(pipeline_steps[f'step_condition_{indication}'])
        steps.append(pipeline_steps[f'register_{indication}'])
        
        
        
        print("=====================>pipeline_name : ", train_pipeline_name)
        print(steps)
        pipeline = Pipeline(
                                name = train_pipeline_name,
                                steps=steps,
                                parameters = [
                                    model_approval_status
                                ]
                            )
        print("=====================>Training Pipeline created : Definition")
        print(json.dumps(pipeline.definition()))
        print("=====================>Role upsert")
        pipeline.upsert(role_arn=role)
        body = {
            "message": "Sagemaker Training Pipeline Creation Successfully!",
            "input": train_pipeline_name
        }
        print(f'Upserted {body}')
    
    # execution = pipeline.start()
    response = {
        "statusCode": 200,
        "body": json.dumps('Updation of training pipelines successful')
    }
    return response
