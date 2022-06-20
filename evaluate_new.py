import pandas as pd

df_true=pd.read_csv('E:/.csv')
df_emotion_mode_final=pd.read_csv('E:/emotion_mode_final.csv')
df_result=pd.read_csv('E:/result.csv')
y_pred=df_result['prediction']
y_true=df_emotion_mode_final['true']

#confusion matrix
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,precision_score,recall_score,f1_score,roc_auc_score

print(precision_score(y_true,y_pred))
print(recall_score(y_true,y_pred))
print(accuracy_score(y_true,y_pred))
print(confusion_matrix(y_true,y_pred))
print(f1_score(y_true,y_pred))
print(roc_auc_score(y_true,y_pred))

print(classification_report(y_true,y_pred,target_names=['angry','neutral','fear','contempt','sad','surprise','dislike','happy'])) #aihub클래스
print(classification_report(y_true,y_pred,target_names=['0','1','2','3','4','5','6','7'])) #aihub클래스
print(classification_report(y_true,y_pred,target_names=['감정없음','놀람','슬픔','기쁨','분노','평온함(신뢰)','불안'])) #외주 클래스 
print(classification_report(y_true,y_pred,target_names=['슬픔','기쁨','불안','상처','분노','당황']))#감정데이터 클래스

