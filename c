[33mcommit f0ee98cc7438e61d863dcfbc1d2cd00d5bcdaad5[m[33m ([m[1;36mHEAD -> [m[1;32mmain[m[33m, [m[1;31morigin/main[m[33m, [m[1;31morigin/HEAD[m[33m)[m
Author: Dilzhan <dilzhan00@gmail.com>
Date:   Fri Jun 21 12:35:31 2024 +0500

    sdfsdfsdf

[1mdiff --git a/.ipynb_checkpoints/bodyfat-prediction-ridge-0-74-r2-checkpoint.ipynb b/.ipynb_checkpoints/bodyfat-prediction-ridge-0-74-r2-checkpoint.ipynb[m
[1mnew file mode 100644[m
[1mindex 0000000..b00aaa9[m
[1m--- /dev/null[m
[1m+++ b/.ipynb_checkpoints/bodyfat-prediction-ridge-0-74-r2-checkpoint.ipynb[m
[36m@@ -0,0 +1,3111 @@[m
[32m+[m[32m{[m
[32m+[m[32m "cells": [[m
[32m+[m[32m  {[m
[32m+[m[32m   "cell_type": "code",[m
[32m+[m[32m   "execution_count": 1,[m
[32m+[m[32m   "id": "0dee5204",[m
[32m+[m[32m   "metadata": {[m
[32m+[m[32m    "_cell_guid": "b1076dfc-b9ad-4769-8c92-a6c4dae69d19",[m
[32m+[m[32m    "_uuid": "8f2839f25d086af736a60e9eeb907d3b93b6e0e5",[m
[32m+[m[32m    "execution": {[m
[32m+[m[32m     "iopub.execute_input": "2024-05-13T07:14:33.974407Z",[m
[32m+[m[32m     "iopub.status.busy": "2024-05-13T07:14:33.973542Z",[m
[32m+[m[32m     "iopub.status.idle": "2024-05-13T07:14:35.022055Z",[m
[32m+[m[32m     "shell.execute_reply": "2024-05-13T07:14:35.020503Z"[m
[32m+[m[32m    },[m
[32m+[m[32m    "papermill": {[m
[32m+[m[32m     "duration": 1.071613,[m
[32m+[m[32m     "end_time": "2024-05-13T07:14:35.025035",[m
[32m+[m[32m     "exception": false,[m
[32m+[m[32m     "start_time": "2024-05-13T07:14:33.953422",[m
[32m+[m[32m     "status": "completed"[m
[32m+[m[32m    },[m
[32m+[m[32m    "tags": [][m
[32m+[m[32m   },[m
[32m+[m[32m   "outputs": [],[m
[32m+[m[32m   "source": [[m
[32m+[m[32m    "import numpy as np\n",[m
[32m+[m[32m    "import pandas as pd"[m
[32m+[m[32m   ][m
[32m+[m[32m  },[m
[32m+[m[32m  {[m
[32m+[m[32m   "cell_type": "markdown",[m
[32m+[m[32m   "id": "a0131655",[m
[32m+[m[32m   "metadata": {[m
[32m+[m[32m    "papermill": {[m
[32m+[m[32m     "duration": 0.01737,[m
[32m+[m[32m     "end_time": "2024-05-13T07:14:35.060425",[m
[32m+[m[32m     "exception": false,[m
[32m+[m[32m     "start_time": "2024-05-13T07:14:35.043055",[m
[32m+[m[32m     "status": "completed"[m
[32m+[m[32m    },[m
[32m+[m[32m    "tags": [][m
[32m+[m[32m   },[m
[32m+[m[32m   "source": [[m
[32m+[m[32m    "# Data Processing"[m
[32m+[m[32m   ][m
[32m+[m[32m  },[m
[32m+[m[32m  {[m
[32m+[m[32m   "cell_type": "markdown",[m
[32m+[m[32m   "id": "3629ac5d",[m
[32m+[m[32m   "metadata": {[m
[32m+[m[32m    "papermill": {[m
[32m+[m[32m     "duration": 0.018101,[m
[32m+[m[32m     "end_time": "2024-05-13T07:14:35.095946",[m
[32m+[m[32m     "exception": false,[m
[32m+[m[32m     "start_time": "2024-05-13T07:14:35.077845",[m
[32m+[m[32m     "status": "completed"[m
[32m+[m[32m    },[m
[32m+[m[32m    "tags": [][m
[32m+[m[32m   },[m
[32m+[m[32m   "source": [[m
[32m+[m[32m    "## Loading the Data"[m
[32m+[m[32m   ][m
[32m+[m[32m  },[m
[32m+[m[32m  {[m
[32m+[m[32m   "cell_type": "code",[m
[32m+[m[32m   "execution_count": 2,[m
[32m+[m[32m   "id": "69ec6f49",[m
[32m+[m[32m   "metadata": {[m
[32m+[m[32m    "execution": {[m
[32m+[m[32m     "iopub.execute_input": "2024-05-13T07:14:35.133326Z",[m
[32m+[m[32m     "iopub.status.busy": "2024-05-13T07:14:35.132664Z",[m
[32m+[m[32m     "iopub.status.idle": "2024-05-13T07:14:35.190623Z",[m
[32m+[m[32m     "shell.execute_reply": "2024-05-13T07:14:35.189526Z"[m
[32m+[m[32m    },[m
[32m+[m[32m    "papermill": {[m
[32m+[m[32m     "duration": 0.079838,[m
[32m+[m[32m     "end_time": "2024-05-13T07:14:35.193200",[m
[32m+[m[32m     "exception": false,[m
[32m+[m[32m     "start_time": "2024-05-13T07:14:35.113362",[m
[32m+[m[32m     "status": "completed"[m
[32m+[m[32m    },[m
[32m+[m[32m    "tags": [][m
[32m+[m[32m   },[m
[32m+[m[32m   "outputs": [[m
[32m+[m[32m    {[m
[32m+[m[32m     "data": {[m
[32m+[m[32m      "text/html": [[m
[32m+[m[32m       "<div>\n",[m
[32m+[m[32m       "<style scoped>\n",[m
[32m+[m[32m       "    .dataframe tbody tr th:only-of-type {\n",[m
[32m+[m[32m       "        vertical-align: middle;\n",[m
[32m+[m[32m       "    }\n",[m
[32m+[m[32m       "\n",[m
[32m+[m[32m       "    .dataframe tbody tr th {\n",[m
[32m+[m[32m       "        vertical-align: top;\n",[m
[32m+[m[32m       "    }\n",[m
[32m+[m[32m       "\n",[m
[32m+[m[32m       "    .dataframe thead th {\n",[m
[32m+[m[32m       "        text-align: right;\n",[m
[32m+[m[32m       "    }\n",[m
[32m+[m[32m       "</style>\n",[m
[32m+[m[32m       "<table border=\"1\" class=\"dataframe\">\n",[m
[32m+[m[32m       "  <thead>\n",[m
[32m+[m[32m       "    <tr style=\"text-align: right;\">\n",[m
[32m+[m[32m       "      <th></th>\n",[m
[32m+[m[32m       "      <th>Density</th>\n",[m
[32m+[m[32m       "      <th>BodyFat</th>\n",[m
[32m+[m[32m       "      <th>Age</th>\n",[m
[32m+[m[32m       "      <th>Weight</th>\n",[m
[32m+[m[32m       "      <th>Height</th>\n",[m
[32m+[m[32m       "      <th>Neck</th>\n",[m
[32m+[m[32m       "      <th>Chest</th>\n",[m
[32m+[m[32m       "      <th>Abdomen</th>\n",[m
[32m+[m[32m       "      <th>Hip</th>\n",[m
[32m+[m[32m       "      <th>Thigh</th>\n",[m
[32m+[m[32m       "      <th>Knee</th>\n",[m
[32m+[m[32m       "      <th>Ankle</th>\n",[m
[32m+[m[32m       "      <th>Biceps</th>\n",[m
[32m+[m[32m       "      <th>Forearm</th>\n",[m
[32m+[m[32m       "      <th>Wrist</th>\n",[m
[32m+[m[32m       "    </tr>\n",[m
[32m+[m[32m       "  </thead>\n",[m
[32m+[m[32m       "  <tbody>\n",[m
[32m+[m[32m       "    <tr>\n",[m
[32m+[m[32m       "      <th>0</th>\n",[m
[32m+[m[32m       "      <td>1.0708</td>\n",[m
[32m+[m[32m       "      <td>12.3</td>\n",[m
[32m+[m[32m       "      <td>23</td>\n",[m
[32m+[m[32m       "      <td>154.25</td>\n",[m
[32m+[m[32m       "      <td>67.75</td>\n",[m
[32m+[m[32m       "      <td>36.2</td>\n",[m
[32m+[m[32m       "      <td>93.1</td>\n",[m
[32m+[m[32m       "      <td>85.2</td>\n",[m
[32m+[m[32m       "      <td>94.5</td>\n",[m
[32m+[m[32m       "      <td>59.0</td>\n",[m
[32m+[m[32m       "      <td>37.3</td>\n",[m
[32m+[m[32m       "      <td>21.9</td>\n",[m
[32m+[m[32m       "      <td>32.0</td>\n",[m
[32m+[m[32m       "      <td>27.4</td>\n",[m
[32m+[m[32m       "      <td>17.1</td>\n",[m
[32m+[m[32m       "    </tr>\n",[m
[32m+[m[32m       "    <tr>\n",[m
[32m+[m[32m       "      <th>1</th>\n",[m
[32m+[m[32m       "      <td>1.0853</td>\n",[m
[32m+[m[32m       "      <td>6.1</td>\n",[m
[32m+[m[32m       "      <td>22</td>\n",[m
[32m+[m[32m       "      <td>173.25</td>\n",[m
[32m+[m[32m       "      <td>72.25</td>\n",[m
[32m+[m[32m       "      <td>38.5</td>\n",[m
[32m+[m[32m       "      <td>93.6</td>\n",[m
[32m+[m[32m       "      <td>83.0</td>\n",[m
[32m+[m[32m       "      <td>98.7</td>\n",[m
[32m+[m[32m       "      <td>58.7</td>\n",[m
[32m+[m[32m       "      <td>37.3</td>\n",[m
[32m+[m[32m       "      <td>23.4</td>\n",[m
[32m+[m[32m       "      <td>30.5</td>\n",[m
[32m+[m[32m       "      <td>28.9</td>\n",[m
[32m+[m[32m       "      <td>18.2</td>\n",[m
[32m+[m[32m       "    </tr>\n",[m
[32m+[m[32m       "    <tr>\n",[m
[32m+[m[32m       "      <th>2</th>\n",[m
[32m+[m[32m       "      <td>1.0414</td>\n",[m
[32m+[m[32m       "      <td>25.3</td>\n",[m
[32m+[m[32m       "      <td>22</td>\n",[m
[32m+[m[32m       "      <td>154.00</td>\n",[m
[32m+[m[32m       "      <td>66.25</td>\n",[m
[32m+[m[32m       "      <td>34.0</td>\n",[m
[32m+[m[32m       "      <td>95.8</td>\n",[m
[32m+[m[32m       "      <td>87.9</td>\n",[m
[32m+[m[32m       "      <td>99.2</td>\n",[m
[32m+[m[32m       "      <td>59.6</td>\n",[m
[32m+[m[32m       "      <td>38.9</td>\n",[m
[32m+[m[32m       "      <td>24.0</td>\n",[m
[32m+[m[32m       "      <td>28.8</td>\n",[m
[32m+[m[32m       "      <td>25.2</td>\n",[m
[32m+[m[32m       "      <td>16.6</td>\n",[m
[32m+[m[32m       "    </tr>\n",[m
[32m+[m[32m       "    <tr>\n",[m
[32m+[m[32m       "      <th>3</th>\n",[m
[32m+[m[32m       "      <td>1.0751</td>\n",[m
[32m+[m[32m       "      <td>10.4</td>\n",[m
[32m+[m[32m       "      <td>26</td>\n",[m
[32m+[m[32m       "      <td>184.75</td>\n",[m
[32m+[m[32m       "      <td>72.25</td>\n",[m
[32m+[m[32m       "      <td>37.4</td>\n",[m
[32m+[m[32m       "      <td>101.8</td>\n",[m
[32m+[m[32m       "      <td>86.4</td>\n",[m
[32m+[m[32m       "      <td>101.2</td>\n",[m
[32m+[m[32m       "      <td>60.1</td>\n",[m
[32m+[m[32m       "      <td>37.3</td>\n",[m
[32m+[m[32m       "      <td>22.8</td>\n",[m
[32m+[m[32m       "      <td>32.4</td>\n",[m
[32m+[m[32m       "      <td>29.4</td>\n",[m
[32m+[m[32m       "      <td>18.2</td>\n",[m
[32m+[m[32m       "    </tr>\n",[m
[32m+[m[32m       "    <tr>\n",[m
[32m+[m[32m       "      <th>4</th>\n",[m
[32m+[m[32m       "      <td>1.0340</td>\n",[m
[32m+[m[32m       "      <td>28.7</td>\n",[m
[32m+[m[32m       "      <td>24</td>\n",[m
[32m+[m[32m       "      <td>184.25</td>\n",[m
[32m+[m[32m       "      <td>71.25</td>\n",[m
[32m+[m[32m       "      <td>34.4</td>\n",[m
[32m+[m[32m       "      <td>97.3</td>\n",[m
[32m+[m[32m       "      <td>100.0</td>\n",[m
[32m+[m[32m       "      <td>101.9</td>\n",[m
[32m+[m[32m       "      <td>63.2</td>\n",[m
[32m+[m[32m       "      <td>42.2</td>\n",[m
[32m+[m[32m       "      <td>24.0</td>\n",[m
[32m+[m[32m       "      <td>32.2</td>\n",[m
[32m+[m[32m       "      <td>27.7</td>\n",[m
[32m+[m[32m       "      <td>17.7</td>\n",[m
[32m+[m[32m       "    </tr>\n",[m
[32m+[m[32m       "  </tbody>\n",[m
[32m+[m[32m       "</table>\n",[m
[32m+[m[32m       "</div>"[m
[32m+[m[32m      ],[m
[32m+[m[32m      "text/plain": [[m
[32m+[m[32m       "   Density  BodyFat  Age  Weight  Height  Neck  Chest  Abdomen    Hip  Thigh  \\\n",[m
[32m+[m[32m       "0   1.0708     12.3   23  154.25   67.75  36.2   93.1     85.2   94.5   59.0   \n",[m
[32m+[m[32m       "1   1.0853      6.1   22  173.25   72.25  38.5   93.6     83.0   98.7   58.7   \n",[m
[32m+[m[32m       "2   1.0414     25.3   22  154.00   66.25  34.0   95.8     87.9   99.2   59.6   \n",[m
[32m+[m[32m       "3   1.0751     10.4   26  184.75   72.25  37.4  101.8     86.4  101.2   60.1   \n",[m
[32m+[m[32m       "4   1.0340     28.7   24  184.25   71.25  34.4   97.3    100.0  101.9   63.2   \n",[m
[32m+[m[32m       "\n",[m
[32m+[m[32m       "   Knee  Ankle  Biceps  Forearm  Wrist  \n",[m
[32m+[m[32m       "0  37.3   21.9    32.0     27.4   17.1  \n",[m
[32m+[m[32m       "1  37.3   23.4    30.5     28.9   18.2  \n",[m
[32m+[m[32m       "2  38.9   24.0    28.8     25.2   16.6  \n",[m
[32m+[m[32m       "3  37.3   22.8    32.4     29.4   18.2  \n",[m
[32m+[m[32m       "4  42.2   24.0    32.2     27.7   17.7  "[m
[32m+[m[32m      ][m
[32m+[m[32m     },[m
[32m+[m[32m     "execution_count": 2,[m
[32m+[m[32m     "metadata": {},[m
[32m+[m[32m     "output_type": "execute_result"[m
[32m+[m[32m    }[m
[32m+[m[32m   ],[m
[32m+[m[32m   "source": [[m
[32m+[m[32m    "df = pd.read_csv('/kaggle/input/body-fat-prediction-dataset/bodyfat.csv')\n",[m
[32m+[m[32m    "\n",[m
[32m+[m[32m    "df.head()"[m
[32m+[m[32m   ][m
[32m+[m[32m  },[m
[32m+[m[32m  {[m
[32m+[m[32m   "cell_type": "code",[m
[32m+[m[32m   "execution_count": 3,[m
[32m+[m[32m   "id": "fb97f3ea",[m
[32m+[m[32m   "metadata": {[m
[32m+[m[32m    "execution": {[m
[32m+[m[32m     "iopub.execute_input": "2024-05-13T07:14:35.230940Z",[m
[32m+[m[32m     "iopub.status.busy": "2024-05-13T07:14:35.230411Z",[m
[32m+[m[32m     "iopub.status.idle": "2024-05-13T07:14:35.240273Z",[m
[32m+[m[32m     "shell.execute_reply": "2024-05-13T07:14:35.238643Z"[m
[32m+[m[32m    },[m
[32m+[m[32m    "papermill": {[m
[32m+[m[32m     "duration": 0.032071,[m
[32m+[m[32m     "end_time": "2024-05-13T07:14:35.243027",[m
[32m+[m[32m     "exception": false,[m
[32m+[m[32m     "start_time": "2024-05-13T07:14:35.210956",[m
[32m+[m[32m     "status": "completed"[m
[32m+[m[32m    },[m
[32m+[m[32m    "tags": [][m
[32m+[m[32m   },[m
[32m+[m[32m   "outputs": [[m
[32m+[m[32m    {[m
[32m+[m[32m     "data": {[m
[32m+[m[32m      "text/plain": [[m
[32m+[m[32m       "(252, 15)"[m
[32m+[m[32m      ][m
[32m+[m[32m     },[m
[32m+[m[32m     "execution_count": 3,[m
[32m+[m[32m     "metadata": {},[m
[32m+[m[32m     "output_type": "execute_result"[m
[32m+[m[32m    }[m
[32m+[m[32m   ],[m
[32m+[m[32m   "source": [[m
[32m+[m[32m    "df.shape"[m
[32m+[m[32m   ][m
[32m+[m[32m  },[m
[32m+[m[32m  {[m
[32m+[m[32m   "cell_type": "code",[m
[32m+[m[32m   "execution_count": 4,[m
[32m+[m[32m   "id": "cf407a42",[m
[32m+[m[32m   "metadata": {[m
[32m+[m[32m    "execution": {[m
[32m+[m[32m     "iopub.execute_input": "2024-05-13T07:14:35.280899Z",[m
[32m+[m[32m     "iopub.status.busy": "2024-05-13T07:14:35.280500Z",[m
[32m+[m[32m     "iopub.status.idle": "2024-05-13T07:14:35.340994Z",[m
[32m+[m[32m     "shell.execute_reply": "2024-05-13T07:14:35.339605Z"[m
[32m+[m[32m    },[m
[32m+[m[32m    "papermill": {[m
[32m+[m[32m     "duration": 0.082452,[m
[32m+[m[32m     "end_time": "2024-05-13T07:14:35.343610",[m
[32m+[m[32m     "exception": false,[m
[32m+[m[32m     "start_time": "2024-05-13T07:14:35.261158",[m
[32m+[m[32m     "status": "completed"[m
[32m+[m[32m    },[m
[32m+[m[32m    "tags": [][m
[32m+[m[32m   },[m
[32m+[m[32m   "outputs": [[m
[32m+[m[32m    {[m
[32m+[m[32m     "data": {[m
[32m+[m[32m      "text/html": [[m
[32m+[m[32m       "<div>\n",[m
[32m+[m[32m       "<style scoped>\n",[m
[32m+[m[32m       "    .dataframe tbody tr th:only-of-type {\n",[m
[32m+[m[32m       "        vertical-align: middle;\n",[m
[32m+[m[32m       "    }\n",[m
[32m+[m[32m       "\n",[m
[32m+[m[32m       "    .dataframe tbody tr th {\n",[m
[32m+[m[32m       "        vertical-align: top;\n",[m
[32m+[m[32m       "    }\n",[m
[32m+[m[32m       "\n",[m
[32m+[m[32m       "    .dataframe thead th {\n",[m
[32m+[m[32m       "        text-align: right;\n",[m
[32m+[m[32m       "    }\n",[m
[32m+[m[32m       "</style>\n",[m
[32m+[m[32m       "<table border=\"1\" class=\"dataframe\">\n",[m
[32m+[m[32m       "  <thead>\n",[m
[32m+[m[32m       "    <tr style=\"text-align: right;\">\n",[m
[32m+[m[32m       "      <th></th>\n",[m
[32m+[m[32m       "      <th>Density</th>\n",[m
[32m+[m[32m       "      <th>BodyFat</th>\n",[m
[32m+[m[32m       "      <th>Age</th>\n",[m
[32m+[m[32m       "      <th>Weight</th>\n",[m
[32m+[m[32m       "      <th>Height</th>\n",[m
[32m+[m[32m       "      <th>Neck</th>\n",[m
[32m+[m[32m       "      <th>Chest</th>\n",[m
[32m+[m[32m       "      <th>Abdomen</th>\n",[m
[32m+[m[32m       "      <th>Hip</th>\n",[m
[32m+[m[32m       "      <th>Thigh</th>\n",[m
[32m+[m[32m       "      <th>Knee</th>\n",[m
[32m+[m[32m       "      <th>Ankle</th>\n",[m
[32m+[m[32m       "      <th>Biceps</th>\n",[m
[32m+[m[32m       "      <th>Forearm</th>\n",[m
[32m+[m[32m       "      <th>Wrist</th>\n",[m
[32m+[m[32m       "    </tr>\n",[m
[32m+[m[32m       "  </thead>\n",[m
[32m+[m[32m       "  <tbody>\n",[m
[32m+[m[32m       "    <tr>\n",[m
[32m+[m[32m       "      <th>count</th>\n",[m
[32m+[m[32m       "      <td>252.000000</td>\n",[m
[32m+[m[32m       "      <td>252.000000</td>\n",[m
[32m+[m[32m       "      <td>252.000000</td>\n",[m
[32m+[m[32m       "      <td>252.000000</td>\n",[m
[32m+[m[32m       "      <td>252.000000</td>\n",[m
[32m+[m[32m       "      <td>252.000000</td>\n",[m
[32m+[m[32m       "      <td>252.000000</td>\n",[m
[32m+[m[32m       "      <td>252.000000</td>\n",[m
[32m+[m[32m       "      <td>252.000000</td>\n",[m
[32m+[m[32m       "      <td>252.000000</td>\n",[m
[32m+[m[32m       "      <td>252.000000</td>\n",[m
[32m+[m[32m       "      <td>252.000000</td>\n",[m
[32m+[m[32m       "      <td>252.000000</td>\n",[m
[32m+[m[32m       "      <td>252.000000</td>\n",[m
[32m+[m[32m       "      <td>252.000000</td>\n",[m
[32m+[m[32m       "    </tr>\n",[m
[32m+[m[32m       "    <tr>\n",[m
[32m+[m[32m       "      <th>mean</th>\n",[m
[32m+[m[32m       "      <td>1.055574</td>\n",[m
[32m+[m[32m       "      <td>19.150794</td>\n",[m
[32m+[m[32m       "      <td>44.884921</td>\n",[m
[32m+[m[32m       "      <td>178.924405</td>\n",[m
[32m+[m[32m       "      <td>70.148810</td>\n",[m
[32m+[m[32m       "      <td>37.992063</td>\n",[m
[32m+[m[32m       "      <td>100.824206</td>\n",[m
[32m+[m[32m       "      <td>92.555952</td>\n",[m
[32m+[m[32m       "      <td>99.904762</td>\n",[m
[32m+[m[32m       "      <td>59.405952</td>\n",[m
[32m+[m[32m       "      <td>38.590476</td>\n",[m
[32m+[m[32m       "      <td>23.102381</td>\n",[m
[32m+[m[32m       "      <td>32.273413</td>\n",[m
[32m+[m[32m       "      <td>28.663889</td>\n",[m
[32m+[m[32m       "      <td>18.229762</td>\n",[m
[32m+[m[32m       "    </tr>\n",[m
[32m+[m[32m       "    <tr>\n",[m
[32m+[m[32m       "      <th>std</th>\n",[m
[32m+[m[32m       "      <td>0.019031</td>\n",[m
[32m+[m[32m       "      <td>8.368740</td>\n",[m
[32m+[m[32m       "      <td>12.602040</td>\n",[m
[32m+[m[32m       "      <td>29.389160</td>\n",[m
[32m+[m[32m       "      <td>3.662856</td>\n",[m
[32m+[m[32m       "      <td>2.430913</td>\n",[m
[32m+[m[32m       "      <td>8.430476</td>\n",[m
[32m+[m[32m       "      <td>10.783077</td>\n",[m
[32m+[m[32m       "      <td>7.164058</td>\n",[m
[32m+[m[32m       "      <td>5.249952</td>\n",[m
[32m+[m[32m       "      <td>2.411805</td>\n",[m
[32m+[m[32m       "      <td>1.694893</td>\n",[m
[32m+[m[32m       "      <td>3.021274</td>\n",[m
[32m+[m[32m       "      <td>2.020691</td>\n",[m
[32m+[m[32m       "      <td>0.933585</td>\n",[m
[32m+[m[32m       "    </tr>\n",[m
[32m+[m[32m       "    <tr>\n",[m
[32m+[m[32m       "      <th>min</th>\n",[m
[32m+[m[32m       "      <td>0.995000</td>\n",[m
[32m+[m[32m       "      <td>0.000000</td>\n",[m
[32m+[m[32m       "      <td>22.000000</td>\n",[m
[32m+[m[32m       "      <td>118.500000</td>\n",[m
[32m+[m[32m       "      <td>29.500000</td>\n",[m
[32m+[m[32m       "      <td>31.100000</td>\n",[m
[32m+[m[32m       "      <td>79.300000</td>\n",[m
[32m+[m[32m       "      <td>69.400000</td>\n",[m
[32m+[m[32m       "      <td>85.000000</td>\n",[m
[32m+[m[32m       "      <td>47.200000</td>\n",[m
[32m+[m[32m       "      <td>33.000000</td>\n",[m
[32m+[m[32m       "      <td>19.100000</td>\n",[m
[32m+[m[32m       "      <td>24.800000</td>\n",[m
[32m+[m[32m       "      <td>21.000000</td>\n",[m
[32m+[m[32m       "      <td>15.800000</td>\n",[m
[32m+[m[32m       "    </tr>\n",[m
[32m+[m[32m       "    <tr>\n",[m
[32m+[m[32m       "      <th>25%</th>\n",[m
[32m+[m[32m       "      <td>1.041400</td>\n",[m
[32m+[m[32m       "      <td>12.475000</td>\n",[m
[32m+[m[32m       "      <td>35.750000</td>\n",[m
[32m+[m[32m       "      <td>159.000000</td>\n",[m
[32m+[m[32m       "      <td>68.250000</td>\n",[m
[32m+[m[32m       "      <td>36.400000</td>\n",[m
[32m+[m[32m       "      <td>94.350000</td>\n",[m
[32m+[m[32m       "      <td>84.575000</td>\n",[m
[32m+[m[32m       "      <td>95.500000</td>\n",[m
[32m+[m[32m       "      <td>56.000000</td>\n",[m
[32m+[m[32m       "      <td>36.975000</td>\n",[m
[32m+[m[32m       "      <td>22.000000</td>\n",[m
[32m+[m[32m       "      <td>30.200000</td>\n",[m
[32m+[m[32m       "      <td>27.300000</td>\n",[m
[32m+[m[32m       "      <td>17.600000</td>\n",[m
[32m+[m[32m       "    </tr>\n",[m
[32m+[m[32m       "    <tr>\n",[m
[32m+[m[32m       "      <th>50%</th>\n",[m
[32m+[m[32m       "      <td>1.054900</td>\n",[m
[32m+[m[32m       "      <td>19.200000</td>\n",[m
[32m+[m[32m       "      <td>43.000000</td>\n",[m
[32m+[m[32m       "      <td>176.500000</td>\n",[m
[32m+[m[32m       "      <td>70.000000</td>\n",[m
[32m+[m[32m       "      <td>38.000000</td>\n",[m
[32m+[m[32m       "      <td>99.650000</td>\n",[m
[32m+[m[32m       "      <td>90.950000</td>\n",[m
[32m+[m[32m       "      <td>99.300000</td>\n",[m
[32m+[m[32m       "      <td>59.000000</td>\n",[m
[32m+[m[32m       "      <td>38.500000</td>\n",[m
[32m+[m[32m       "      <td>22.800000</td>\n",[m
[32m+[m[32m       "      <td>32.050000</td>\n",[m
[32m+[m[32m       "      <td>28.700000</td>\n",[m
[32m+[m[32m       "      <td>18.300000</td>\n",[m
[32m+[m[32m       "    </tr>\n",[m
[32m+[m[32m       "    <tr>\n",[m
[32m+[m[32m       "      <th>75%</th>\n",[m
[32m+[m[32m       "      <td>1.070400</td>\n",[m
[32m+[m[32m       "      <td>25.300000</td>\n",[m
[32m+[m[32m       "      <td>54.000000</td>\n",[m
[32m+[m[32m       "      <td>197.000000</td>\n",[m
[32m+[m[32m       "      <td>72.250000</td>\n",[m
[32m+[m[32m       "      <td>39.425000</td>\n",[m
[32m+[m[32m       "      <td>105.375000</td>\n",[m
[32m+[m[32m       "      <td>99.325000</td>\n",[m
[32m+[m[32m       "      <td>103.525000</td>\n",[m
[32m+[m[32m       "      <td>62.350000</td>\n",[m
[32m+[m[32m       "      <td>39.925000</td>\n",[m
[32m+[m[32m       "      <td>24.000000</td>\n",[m
[32m+[m[32m       "      <td>34.325000</td>\n",[m
[32m+[m[32m       "      <td>30.000000</td>\n",[m
[32m+[m[32m       "      <td>18.800000</td>\n",[m
[32m+[m[32m       "    </tr>\n",[m
[32m+[m[32m       "    <tr>\n",[m
[32m+[m[32m       "      <th>max</th>\n",[m
[32m+[m[32m       "      <td>1.108900</td>\n",[m
[32m+[m[32m       "      <td>47.500000</td>\n",[m
[32m+[m[32m       "      <td>81.000000</td>\n",[m
[32m+[m[32m       "      <td>363.150000</td>\n",[m
[32m+[m[32m       "      <td>77.750000</td>\n",[m
[32m+[m[32m       "      <td>51.200000</td>\n",[m
[32m+[m[32m       "      <td>136.200000</td>\n",[m
[32m+[m[32m       "      <td>148.100000</td>\n",[m
[32m+[m[32m       "      <td>147.700000</td>\n",[m
[32m+[m[32m       "      <td>87.300000</td>\n",[m
[32m+[m[32m       "      <td>49.100000</td>\n",[m
[32m+[m[32m       "      <td>33.900000</td>\n",[m
[32m+[m[32m       "      <td>45.000000</td>\n",[m
[32m+[m[32m       "      <td>34.900000</td>\n",[m
[32m+[m[32m       "      <td>21.400000</td>\n",[m
[32m+[m[32m       "    </tr>\n",[m
[32m+[m[32m       "  </tbody>\n",[m
[32m+[m[32m       "</table>\n",[m
[32m+[m[32m       "</div>"[m
[32m+[m[32m      ],[m
[32m+[m[32m      "text/plain": [[m
[32m+[m[32m       "          Density     BodyFat         Age      Weight      Height        Neck  \\\n",[m
[32m+[m[32m       "count  252.000000  252.000000  252.000000  252.000000  252.000000  252.000000   \n",[m
[32m+[m[32m       "mean     1.055574   19.150794   44.884921  178.924405   70.148810   37.992063   \n",[m
[32m+[m[32m       "std      0.019031    8.368740   12.602040   29.389160    3.662856    2.430913   \n",[m
[32m+[m[32m       "min      0.995000    0.000000   22.000000  118.500000   29.500000   31.100000   \n",[m
[32m+[m[32m       "25%      1.041400   12.475000   35.750000  159.000000   68.250000   36.400000   \n",[m
[32m+[m[32m       "50%      1.054900   19.200000   43.000000  176.500000   70.000000   38.000000   \n",[m
[32m+[m[32m       "75%      1.070400   25.300000   54.000000  197.000000   72.250000   39.425000   \n",[m
[32m+[m[32m       "max      1.108900   47.500000   81.000000  363.150000   77.750000   51.200000   \n",[m
[32m+[m[32m       "\n",[m
[32m+[m[32m       "            Chest     Abdomen         Hip       Thigh        Knee       Ankle  \\\n",[m
[32m+[m[32m       "count  252.000000  252.000000  252.000000  252.000000  252.000000  252.000000   \n",[m
[32m+[m[32m       "mean   100.824206   92.555952   99.904762   59.405952   38.590476   23.102381   \n",[m
[32m+[m[32m       "std      8.430476   10.783077    7.164058    5.249952    2.411805    1.694893   \n",[m
[32m+[m[32m       "min     79.300000   69.400000   85.000000   47.200000   33.000000   19.100000   \n",[m
[32m+[m[32m       "25%     94.350000   84.575000   95.500000   56.000000   36.975000   22.000000   \n",[m
[32m+[m[32m       "50%     99.650000   90.950000   99.300000   59.000000   38.500000   22.800000   \n",[m
[32m+[m[32m       "75%    105.375000   99.325000  103.525000   62.350000   39.925000   24.000000   \n",[m
[32m+[m[32m       "max    136.200000  148.100000  147.700000   87.300000   49.100000   33.900000   \n",[m
[32m+[m[32m       "\n",[m
[32m+[m[32m       "           Biceps     Forearm       Wrist  \n",[m
[32m+[m[32m       "count  252.000000  252.000000  252.000000  \n",[m
[32m+[m[32m       "mean    32.273413   28.663889   18.229762  \n",[m
[32m+[m[32m       "std      3.021274    2.020691    0.933585  \n",[m
[32m+[m[32m       "min     24.800000   21.000000   15.800000  \n",[m
[32m+[m[32m       "25%     30.200000   27.300000   17.600000  \n",[m
[32m+[m[32m       "50%     32.050000   28.700000   18.300000  \n",[m
[32m+[m[32m       "75%     34.325000   30.000000   18.800000  \n",[m
[32m+[m[32m       "max     45.000000   34.900000   21.400000  "[m
[32m+[m[32m      ][m
[32m+[m[32m     },[m
[32m+[m[32m     "execution_count": 4,[m
[32m+[m[32m     "metadata": {},[m
[32m+[m[32m     "output_type": "execute_result"[m
[32m+[m[32m    }[m
[32m+[m[32m   ],[m
[32m+[m[32m   "source": [[m
[32m+[m[32m    "df.describe()"[m
[32m+[m[32m   ][m
[32m+[m[32m  },[m
[32m+[m[32m  {[m
[32m+[m[32m   "cell_type": "code",[m
[32m+[m[32m   "execution_count": 5,[m
[32m+[m[32m   "id": "19d04a51",[m
[32m+[m[32m   "metadata": {[m
[32m+[m[32m    "execution": {[m
[32m+[m[32m     "iopub.execute_input": "2024-05-13T07:14:35.382543Z",[m
[32m+[m[32m     "iopub.status.busy": "2024-05-13T07:14:35.382081Z",[m
[32m+[m[32m     "iopub.status.idle": "2024-05-13T07:14:35.391581Z",[m
[32m+[m[32m     "shell.execute_reply": "2024-05-13T07:14:35.390609Z"[m
[32m+[m[32m    },[m
[32m+[m[32m    "papermill": {[m
[32m+[m[32m     "duration": 0.031709,[m
[32m+[m[32m     "end_time": "2024-05-13T07:14:35.393780",[m
[32m+[m[32m     "exception": false,[m
[32m+[m[32m     "start_time": "2024-05-13T07:14:35.362071",[m
[32m+[m[32m     "status": "completed"[m
[32m+[m[32m    },[m
[32m+[m[32m    "tags": [][m
[32m+[m[32m   },[m
[32m+[m[32m   "outputs": [[m
[32m+[m[32m    {[m
[32m+[m[32m     "data": {[m
[32m+[m[32m      "text/plain": [[m
[32m+[m[32m       "Density    0\n",[m
[32m+[m[32m       "BodyFat    0\n",[m
[32m+[m[32m       "Age        0\n",[m
[32m+[m[32m       "Weight     0\n",[m
[32m+[m[32m       "Height     0\n",[m
[32m+[m[32m       "Neck       0\n",[m
[32m+[m[32m       "Chest      0\n",[m
[32m+[m[32m       "Abdomen    0\n",[m
[32m+[m[32m       "Hip        0\n",[m
[32m+[m[32m       "Thigh      0\n",[m
[32m+[m[32m       "Knee       0\n",[m
[32m+[m[32m       "Ankle      0\n",[m
[32m+[m[32m       "Biceps     0\n",[m
[32m+[m[32m       "Forearm    0\n",[m
[32m+[m[32m       "Wrist      0\n",[m
[32m+[m[32m       "dtype: int64"[m
[32m+[m[32m      ][m
[32m+[m[32m     },[m
[32m+[m[32m     "execution_count": 5,[m
[32m+[m[32m     "metadata": {},[m
[32m+[m[32m     "output_type": "execute_result"[m
[32m+[m[32m    }[m
[32m+[m[32m   ],[m
[32m+[m[32m   "source": [[m
[32m+[m[32m    "df.isna().sum()"[m
[32m+[m[32m   ][m
[32m+[m[32m  },[m
[32m+[m[32m  {[m
[32m+[m[32m   "cell_type": "markdown",[m
[32m+[m[32m   "id": "0941ee4e",[m
[32m+[m[32m   "metadata": {[m
[32m+[m[32m    "papermill": {[m
[32m+[m[32m     "duration": 0.018481,[m
[32m+[m[32m     "end_time": "2024-05-13T07:14:35.430659",[m
[32m+[m[32m     "exception": false,[m
[32m+[m[32m     "start_time": "2024-05-13T07:14:35.412178",[m
[32m+[m[32m     "status": "completed"[m
[32m+[m[32m    },[m
[32m+[m[32m    "tags": [][m
[32m+[m[32m   },[m
[32m+[m[32m   "source": [[m
[32m+[m[32m    "## Correlation between the Body Fat Percentage and Other Features"[m
[32m+[m[32m   ][m
[32m+[m[32m  },[m
[32m+[m[32m  {[m
[32m+[m[32m   "cell_type": "code",[m
[32m+[m[32m   "execution_count": 6,[m
[32m+[m[32m   "id": "68dd31ed",[m
[32m+[m[32m   "metadata": {[m
[32m+[m[32m    "execution": {[m
[32m+[m[32m     "iopub.execute_input": "2024-05-13T07:14:35.470228Z",[m
[32m+[m[32m     "iopub.status.busy": "2024-05-13T07:14:35.469091Z",[m
[32m+[m[32m     "iopub.status.idle": "2024-05-13T07:14:35.483790Z",[m
[32m+[m[32m     "shell.execute_reply": "2024-05-13T07:14:35.482831Z"[m
[32m+[m[32m    },[m
[32m+[m[32m    "papermill": {[m
[32m+[m[32m     "duration": 0.037051,[m
[32m+[m[32m     "end_time": "2024-05-13T07:14:35.486094",[m
[32m+[m[32m     "exception": false,[m
[32m+[m[32m     "start_time": "2024-05-13T07:14:35.449043",[m
[32m+[m[32m     "status": "completed"[m
[32m+[m[32m    },[m
[32m+[m[32m    "tags": [][m
[32m+[m[32m   },[m
[32m+[m[32m   "outputs": [[m
[32m+[m[32m    {[m
[32m+[m[32m     "data": {[m
[32m+[m[32m      "text/plain": [[m
[32m+[m[32m       "Density   -0.987782\n",[m
[32m+[m[32m       "BodyFat    1.000000\n",[m
[32m+[m[32m       "Age        0.291458\n",[m
[32m+[m[32m       "Weight     0.612414\n",[m
[32m+[m[32m       "Height    -0.089495\n",[m
[32m+[m[32m       "Neck       0.490592\n",[m
[32m+[m[32m       "Chest      0.702620\n",[m
[32m+[m[32m       "Abdomen    0.813432\n",[m
[32m+[m[32m       "Hip        0.625201\n",[m
[32m+[m[32m       "Thigh      0.559608\n",[m
[32m+[m[32m       "Knee       0.508665\n",[m
[32m+[m[32m       "Ankle      0.265970\n",[m
[32m+[m[32m       "Biceps     0.493271\n",[m
[32m+[m[32m       "Forearm    0.361387\n",[m
[32m+[m[32m       "Wrist      0.346575\n",[m
[32m+[m[32m       "Name: BodyFat, dtype: float64"[m
[32m+[m[32m      ][m
[32m+[m[32m     },[m
[32m+[m[32m     "execution_count": 6,[m
[32m+[m[32m     "metadata": {},[m
[32m+[m[32m     "output_type": "execute_result"[m
[32m+[m[32m    }[m
[32m+[m[32m   ],[m
[32m+[m[32m   "source": [[m
[32m+[m[32m    "corr = df.corr()\n",[m
[32m+[m[32m    "\n",[m
[32m+[m[32m    "corr['BodyFat']"[m
[32m+[m[32m   ][m
[32m+[m[32m  },[m
[32m+[m[32m  {[m
[32m+[m[32m   "cell_type": "code",[m
[32m+[m[32m   "execution_count": 7,[m
[32m+[m[32m   "id": "d52e96b1",[m
[32m+[m[32m   "metadata": {[m
[32m+[m[32m    "execution": {[m
[32m+[m[32m     "iopub.execute_input": "2024-05-13T07:14:35.526620Z",[m
[32m+[m[32m     "iopub.status.busy": "2024-05-13T07:14:35.525743Z",[m
[32m+[m[32m     "iopub.status.idle": "2024-05-13T07:14:35.848368Z",[m
[32m+[m[32m     "shell.execute_reply": "2024-05-13T07:14:35.847243Z"[m
[32m+[m[32m    },[m
[32m+[m[32m    "papermill": {[m
[32m+[m[32m     "duration": 0.346027,[m
[32m+[m[32m     "end_time": "2024-05-13T07:14:35.850815",[m
[32m+[m[32m     "exception": false,[m
[32m+[m[32m     "start_time": "2024-05-13T07:14:35.504788",[m
[32m+[m[32m     "status": "completed"[m
[32m+[m[32m    },[m
[32m+[m[32m    "tags": [][m
[32m+[m[