import os
import pickle


param_pkl_file = "TestSamples/sign/results/msasl_test_v0_f0/msasl_test_v0_f0_param.pkl"

param_dict = pickle.load(open(param_pkl_file, "rb"))

print("param keys", param_dict.keys())

for k, v in param_dict.items():
    print(k, v.shape)


prediction_pkl_file = "TestSamples/sign/results/msasl_test_v0_f0/msasl_test_v0_f0_prediction.pkl"

prediction_dict = pickle.load(open(prediction_pkl_file, "rb"))

print("prediction keys", prediction_dict.keys())

for k, v in prediction_dict.items():
    print(k, v.shape)

print("smplx_kpt3d", prediction_dict["smplx_kpt3d"][0])
print("joints", prediction_dict["joints"][0])
