import boto3
import time
import json
import os


vector_store_name = 'bedrock-workshop-rag'
index_name = "bedrock-workshop-rag-index"
encryption_policy_name = "bedrock-workshop-rag-sp"
network_policy_name = "bedrock-workshop-rag-np"
access_policy_name = 'bedrock-workshop-rag-ap'
identity = boto3.client('sts').get_caller_identity()['Arn']

aoss_client = boto3.client('opensearchserverless')

security_policy = aoss_client.create_security_policy(
    name = encryption_policy_name,
    policy = json.dumps(
        {
            'Rules': [{'Resource': ['collection/' + vector_store_name],
            'ResourceType': 'collection'}],
            'AWSOwnedKey': True
        }),
    type = 'encryption'
)

network_policy = aoss_client.create_security_policy(
    name = network_policy_name,
    policy = json.dumps(
        [
            {'Rules': [{'Resource': ['collection/' + vector_store_name],
            'ResourceType': 'collection'}],
            'AllowFromPublic': True}
        ]),
    type = 'network'
)

collection = aoss_client.create_collection(name=vector_store_name,type='VECTORSEARCH')

while True:
    status = aoss_client.list_collections(collectionFilters={'name':vector_store_name})['collectionSummaries'][0]['status']
    if status in ('ACTIVE', 'FAILED'): break
    time.sleep(10)

access_policy = aoss_client.create_access_policy(
    name = access_policy_name,
    policy = json.dumps(
        [
            {
                'Rules': [
                    {
                        'Resource': ['collection/' + vector_store_name],
                        'Permission': [
                            'aoss:CreateCollectionItems',
                            'aoss:DeleteCollectionItems',
                            'aoss:UpdateCollectionItems',
                            'aoss:DescribeCollectionItems'],
                        'ResourceType': 'collection'
                    },
                    {
                        'Resource': ['index/' + vector_store_name + '/*'],
                        'Permission': [
                            'aoss:CreateIndex',
                            'aoss:DeleteIndex',
                            'aoss:UpdateIndex',
                            'aoss:DescribeIndex',
                            'aoss:ReadDocument',
                            'aoss:WriteDocument'],
                        'ResourceType': 'index'
                    }],
                'Principal': [identity],
                'Description': 'Easy data policy'}
        ]),
    type = 'data'
)

host = collection['createCollectionDetail']['id'] + '.' + os.environ.get("AWS_DEFAULT_REGION", None) + '.aoss.amazonaws.com: