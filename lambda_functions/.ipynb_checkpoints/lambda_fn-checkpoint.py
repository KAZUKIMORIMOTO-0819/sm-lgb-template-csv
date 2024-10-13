import json
import boto3
from logging import getLogger, INFO

### ログの設定
logger = getLogger(__name__)
logger.setLevel(INFO)

### SageMaker runtimeクライアントとS3クライアントの初期化
sagemaker_runtime = boto3.client('sagemaker-runtime')
s3_client = boto3.client('s3')

### 環境変数からエンドポイント名を取得
ENDPOINT_NAME = "lightgbm-deploy-template"
logger.info(f"Using SageMaker endpoint: {ENDPOINT_NAME}")


def lambda_handler(event, context):
    """
    S3にアップロードされたJSONファイルを処理し、推論結果をS3に保存するLambda関数

    :param event: S3イベント
    :param context: Lambdaコンテキスト
    :return: 成功メッセージ
    """
    logger.info("==== Start lambda functions ====")  # Lambda関数の開始を示すログ
    try:
        ### S3イベントからバケット名とオブジェクトキーを取得
        bucket_name = event['Records'][0]['s3']['bucket']['name']
        file_key = event['Records'][0]['s3']['object']['key']
        
        logger.info(f"Received S3 event. Bucket: {bucket_name}, Key: {file_key}")

        ### JSONファイルをS3からダウンロード
        response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        file_content = response['Body'].read().decode('utf-8')
        input_data = json.loads(file_content)
        
        logger.info("Successfully downloaded JSON file from S3 and parsed input data.")

        ### SageMakerエンドポイントで推論を実行
        logger.info(f"Invoking SageMaker endpoint: {ENDPOINT_NAME}")
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType='application/json',
            Body=json.dumps(input_data)
        )
        
        result = json.loads(response['Body'].read().decode('utf-8'))
        logger.info(f"Inference completed. Result: {result}")

        ### 結果をJSONファイルに変換してS3に保存
        result_file_key = file_key.replace('.json', '_result.json')
        s3_client.put_object(
            Bucket=bucket_name,
            Key=result_file_key,
            Body=json.dumps(result),
            ContentType='application/json'
        )
        
        logger.info(f"Prediction result saved to S3. Bucket: {bucket_name}, Key: {result_file_key}")

        return {
            'statusCode': 200,
            'body': json.dumps(f"Prediction result saved to {result_file_key} in S3 bucket {bucket_name}")
        }
    
    except Exception as e:
        logger.error(f"Error processing the file. Details: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps(f"Error: {str(e)}")
        }

