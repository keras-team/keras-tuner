
from ..engine import hyperparameters as hp_module
from ..protos import kerastuner_pb2
from ..protos import service_pb2
from ..protos import service_pb2_grpc


class OracleServicer(service_pb2_grpc.OracleServicer):

    def __init__(self, oracle):
        self.oracle = oracle

    def GetSpace(self, request, context):
        hps = self.oracle.hyperparameters.to_proto()
        return service_pb2.GetSpaceResponse(
            hyperparameters=hps)



