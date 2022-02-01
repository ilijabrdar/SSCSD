import shutil
import pandas as pd

dest = '.'
src = 'C:\\Users\\ilija\\OneDrive\\Desktop\\SIAP\\data\\long_method\\files\\'
excel = 'C:\\Users\\ilija\\OneDrive\\Desktop\\SIAP\\data\\long_method\\MLCQ_metrics_LM_binary.xlsx'

df = pd.read_excel(excel)
df = df[df['constructor'] == 1]
df['sample_id'] = df['sample_id'].apply(lambda x: str(x) + '.java')
constructors = df['sample_id']

to_copy = ['3700134.java', '3717906.java', '3719020.java', '3762849.java', '3814167.java', '3818939.java', '3823798.java', '3825676.java', '3868603.java', '3875268.java', '3876033.java', '3879673.java', '3924383.java', '3926659.java', '3945199.java', '3981428.java', '4038375.java', '4043468.java', '4073835.java', '4095482.java', '4106696.java', '4132083.java', '4172212.java', '4188059.java', '4217323.java', '4276035.java', '4282272.java', '4291944.java', '4298894.java', '4332339.java', '4352708.java', '4371297.java', '4383044.java', '4404036.java', '4417713.java', '4437678.java', '4452266.java', '4457261.java', '4501314.java', '4527035.java', '4535026.java', '4536178.java', '4541855.java', '4600851.java', '4666091.java', '4737348.java', '4767234.java', '4852058.java', '4927730.java', '4944991.java', '5004854.java', '5014423.java', '5066014.java', '5161273.java', '5180986.java', '5202532.java', '5235274.java', '5262454.java', '5294252.java', '5297721.java', '5366640.java', '5414837.java', '5423009.java', '5450234.java', '5497852.java', '5515872.java', '5534322.java', '5535647.java', '5555115.java', '5661062.java', '5690366.java', '5698693.java', '5771277.java', '5784280.java', '5798489.java', '5843392.java', '5882133.java', '5907756.java', '5939954.java', '5958136.java', '6009570.java', '6016779.java', '6021504.java', '6029853.java', '6164700.java', '6179494.java', '6186813.java', '6236024.java', '6241200.java', '6247819.java', '6286234.java', '6296221.java', '6350174.java', '6417488.java', '6458156.java', '6459500.java', '6471245.java', '6471435.java', '6488574.java', '6491821.java', '6492213.java', '6502815.java', '6513603.java', '6530288.java', '6534806.java', '6553535.java', '6567017.java', '6633402.java', '6639778.java', '6660510.java', '6666668.java', '6679515.java', '6840403.java', '6850124.java', '6888470.java', '6914478.java', '6933669.java', '6986522.java', '7009087.java', '7023035.java', '7026916.java', '7030257.java', '7031904.java', '7043418.java', '7070559.java', '7075067.java', '7076341.java', '7078266.java', '7110712.java', '7117242.java', '7138422.java', '7144792.java', '7232869.java', '7273237.java', '7273316.java', '7273813.java', '7278060.java', '7294422.java', '7301073.java', '7320020.java', '7320330.java', '7344807.java', '7357208.java', '7357470.java', '7373962.java', '7374084.java', '7379441.java', '7424019.java', '7445434.java', '7451343.java', '7451785.java', '7469146.java', '7473205.java', '7476486.java', '7483452.java', '7483476.java', '7483890.java', '7484147.java', '7485370.java', '7491946.java', '7501278.java', '7513487.java', '7523131.java', '7528643.java', '7542297.java', '7588956.java', '7589426.java', '7674551.java', '7706698.java', '7717190.java', '7726305.java', '7727470.java', '7783504.java', '7883024.java', '7893182.java', '7926598.java', '7926909.java', '7929441.java', '7960668.java', '8009118.java', '8009299.java', '8025955.java', '8049607.java', '8132887.java', '8132913.java', '8195306.java', '8201469.java', '8243636.java', '8334400.java', '8421755.java', '8424288.java', '8432784.java', '8432933.java', '8435790.java', '8452328.java', '8455135.java', '8496550.java', '8521588.java', '8574365.java', '8575719.java', '8576209.java', '8615469.java', '8644476.java', '8659626.java', '8684961.java', '8697369.java', '8712093.java', '8719880.java', '8775243.java', '8828539.java', '8902996.java', '8903716.java', '8904717.java', '8935070.java', '8970572.java', '8979235.java', '9005432.java', '9005699.java', '9009658.java', '9087427.java', '9094331.java', '9097530.java', '9121712.java', '9129936.java', '9194605.java', '9209481.java', '9209895.java', '9215037.java', '9279084.java', '9301230.java', '9312894.java', '9329773.java', '9375078.java', '9407577.java', '9471363.java', '9506659.java']
to_copy = set(to_copy).difference(set(constructors))

for file in to_copy:
	shutil.copy(f'{src}{file}', dest)

