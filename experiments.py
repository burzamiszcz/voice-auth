from VoiceVerificator import VoiceVerificator

voice_verificator = VoiceVerificator()
voice_verificator.threshold = -34
username = 'michal'
# usernames = ['dariusz', 'joanna', 'michal', 'maja', 'hubert']
verification_usernames = ['dariusz', 'joanna', 'michal', 'maja', 'hubert']
usernames = ['hubert']
verification_files = ['pom', 'tekst', 'owoce']
fruits = ['banan', 'grejpfrut', 'jablko', 'winogrona']
# fruits = ['banan', 'grejpfrut']

TP = 0
FP = 0
TN = 0
FN = 0
# TP i FN
# for u in usernames:
#     for vf in verification_files:
#         if vf != 'owoce':
#             voice_verificator.username = u + '_' + vf
#             voice_verificator.test_model_for_experiment(verification_username=u + '_' + vf)
#             if voice_verificator.log_likelihood > voice_verificator.threshold:
#                 TP += 1
#             else:
#                 FN += 1
#         else:
#             for f in fruits:
#                 voice_verificator.username = u + '_' + vf
#                 voice_verificator.test_model_for_experiment(verification_username=u + '_' + vf + '_' + f)
#                 if voice_verificator.log_likelihood > voice_verificator.threshold:
#                     TP += 1
#                 else:
#                     FN += 1
#
# print('----------------------------------------')

for u in usernames:
    for vf in verification_files:
        for vu in verification_usernames:
            if u != vu:
                if vf != 'owoce':
                    voice_verificator.username = u + '_' + vf
                    voice_verificator.test_model_for_experiment(verification_username=vu + '_' + vf)
                    if voice_verificator.log_likelihood > voice_verificator.threshold:
                        FP += 1
                    else:
                        TN += 1
                else:
                    for f in fruits:
                        voice_verificator.username = u + '_' + vf
                        voice_verificator.test_model_for_experiment(verification_username=vu + '_' + vf + '_' + f)
                        if voice_verificator.log_likelihood > voice_verificator.threshold:
                            FP += 1
                        else:
                            TN += 1

print('TP = ' + str(TP))
print('FP = ' + str(FP))
print('TN = ' + str(TN))
print('FN = ' + str(FN))






# voice_verificator.username = 'dariusz_tekst'
# voice_verificator.test_model_for_experiment(verification_username='dariusz_tekst')
# voice_verificator.username = 'dariusz_owoce'
# voice_verificator.test_model_for_experiment(verification_username='dariusz_tekst')
