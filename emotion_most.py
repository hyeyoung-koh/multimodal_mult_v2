import pandas as pd
from collections import Counter
df=pd.read_csv('D:/Users/hyeyoung/OneDrive - dongguk.edu/문서/카카오톡 받은 파일/data_final_full.csv',encoding='utf-8-sig') #인코딩 ok
#df=pd.read_csv('D:/Users/hyeyoung/OneDrive - dongguk.edu/문서/카카오톡 받은 파일/predict_output.tsv',sep='\t')
clip_list=list(df['Clip_no'].unique())
clip_list
def modefinder2(nums):
    c=Counter(nums)
    order=c.most_common()
    maximum=order[0][1]
    modes=[]
    for num in order:
        if num[1]==maximum:
            modes.append(num[0])
    return modes #최빈값 2개이상 return

true_most_emotion_list=[]
predict_most_emotion_list=[]

i_list=[]
# for i in range(1,5601):
#     if i==2005 or i==4016 or i==3038 or i==3101 or i==3109 or i==3166 or i==3173 or i==3174 or i==3177 or i==3193 or i==3251 \
#             or i==3270 or i==3273 or i==3274 or i==3288 or i==3289 or i==3290 or i==3291 or i==3295 or i==3296 or \
#             i==3299 or i==3301 or i==3305 or i==3312 or i==3315 or i==3316 or i==3320 or i==3331 or i==3333 or i==3337 or i==3356 \
#             or i==3366 or i==3401 or i==3402 or i==3405 or i==3406 or i==3408 or i==3412 or i==3414 or i==3415 or i==3416 or i==3427 \
#             or i==3429 or i==3435 or i==3437 or i==3440:
#         continue
#df.columns
for i in clip_list:
    condition = (df['Clip_no']==i)
    condition_index=df[df['Clip_no']==i].index
    print(condition_index) #0,1,2,3,4,5,6,7,8,9
    true_emotion_list = []
    predict_emotion_list=[]
    for j in condition_index: #j=condition_index안의 인덱스 값
        true_emotion_list.append(df['y_true'][j])
        predict_emotion_list.append(df['y_pred'][j])
    print('true_emotion_list:', true_emotion_list) #clip의 emotion list를 다 나열한 list
    print('predict_emotion_list:',predict_emotion_list)
    i_list.append(i)
    true_most_emotion = modefinder2(true_emotion_list)
    true_most_emotion_list.append(true_most_emotion) #clip의 true감정
    predict_most_emotion=modefinder2(predict_emotion_list)
    predict_most_emotion_list.append(predict_most_emotion) #clip의 predict감정

#list값 데이터프레임에
df_new=pd.read_csv('D:/종설/new_dataframe.csv')
print(len(true_most_emotion_list)) #1216
#clip_no_list=df['Clip_no'].unique()
#print(len(clip_no_list)) #5554

#f_new['Clip_no_list']=clip_no_list

#df_new['Clip_no']=i
df_new['true_most_emotion']=true_most_emotion_list
df_new['predict_most_emotion']=predict_most_emotion_list
df_new.to_csv('D:/종설/True_Predict_most_emotion_textonly.csv')

# # #====확인===========================================
# correct=0
# for i in range(len(df_new)):
#     print(df_new['true_most_emotion'])
#     if df_new['true_most_emotion'][i]==df_new['predict_most_emotion'][i]:
#         correct+=1
# print(correct) #322
correct=0
for i in range(len(df_new)):
    #print(df_new['true_most_emotion'])
    for x in df_new['true_most_emotion'][i]:
        if x in df_new['predict_most_emotion'][i]:
            correct+=1
            break
accuracy=correct/len(df_new)
    # if df_new['true_most_emotion'][i]==df_new['predict_most_emotion'][i]:
    #     correct+=1
print(correct) #688
print('accuracy=',accuracy)
# #----------------------------
# y_pred=predict_most_emotion_list
# y_true=true_most_emotion_list

# from sklearn.metrics import *
# print(f1_score(y_pred, y_true))
# print(accuracy_score(y_pred, y_true))
