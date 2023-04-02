# Rebuilding rebuilding the gPRC protos
# grpcio-tools==1.30.*
python -m grpc_tools.protoc --python_out=. --grpc_python_out=. --proto_path=. keras_tuner/protos/keras_tuner.proto
python -m grpc_tools.protoc --python_out=. --grpc_python_out=. --proto_path=. keras_tuner/protos/service.proto
