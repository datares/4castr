from TraderTeam.GetData.newOCHLV import concatenate
from TraderTeam.InputTensorOptimization.featureEngineering import momentum, intraday_change

testData = concatenate(num_files=2)  # small subset to allow us to do quick testing
