# this script takes args as input and extracts the minimum rules from the grammar
# the args are the data file and the item condition

# for example, for sap_items_ClassicGP.csv data file, the example row is:
# berkeley_neural_parser_result,item,condition,disambPosition,Sentence,ambiguous,Question,Option1,Option0,Answer,Ambiguity targeted?
# (S (NP (DT The) (NN suspect)) (VP (VBD showed) (SBAR (IN that) (S (NP (DT the) (NN file)) (VP (VBD deserved) (NP (JJ further) (NN investigation)) (PP (IN during) (NP (DT the) (NN murder) (NN trial))))))) (. .)),1,NPS_UAMB,7,The suspect showed that the file deserved further investigation during the murder trial.,0,Who drew attention to the file during the trial?,The suspect,The lawyer,1,X

# if the other arg value is NPS_UAMB, then we will filter the data file to only include the rows where the condition is NPS_UAMB
# then with the filtered data, for each sentence, we will use the disambPosition to narrow the range of words we are interested in 
# for example, if the diambPosition is 7, then we will only consider words from 1 to 7 + 1 (1-indexed), where punctuation does not count as a word
# we iterate through each sentence and aggregate the words into a set
# then we will use the set to extract the minimum rules from the grammar

# first, we need to convert the words set into labels by using the berkeley_parser_sm5.lexicon file in the trained_berkeley_parser_sm5 folder
# here are two sample lines in the berkeley_parser_sm5.lexicon file:
# IN 'til [7.867165838569136E-6, 4.525270376405287E-6, 6.033708985208875E-6, 4.634812504399481E-6, 5.109573540264209E-6, 7.72116999369993E-6, 5.171389466277146E-6, 2.2501944589313282E-5, 1.6814756028623665E-4, 2.8288951615156508E-5, 5.552975042278512E-6, 1.0477714909688043E-5, 1.4638507666702597E-5, 8.175650283951388E-5, 1.0855890642797818E-5, 8.332817863345324E-6, 1.594645288023604E-5, 7.56416381008608E-6, 9.879876122944983E-6, 8.288940634730868E-6, 6.44875943388881E-4, 1.1151417948698996E-5, 7.812195873833272E-6]
# IN Behind [3.223133909240918E-5, 1.8539780014765882E-5, 2.471976875506592E-5, 1.8988567995689777E-5, 2.0933637446215388E-5, 3.163320226923551E-5, 2.1186893842930244E-5, 9.218921034320903E-5, 6.888911641568645E-4, 1.1589825494802335E-4, 2.275022863785421E-5, 4.292661284856924E-5, 5.99731484112373E-5, 3.3495182637578117E-4, 4.447597770759177E-5, 3.4139089433205104E-5, 6.533172690783057E-5, 3.0989956703106274E-5, 4.047730072607117E-5, 3.395932687793082E-5, 0.002642020726448123, 4.5686736575908634E-5, 3.200613021672828E-5]
# we want to find all possible labels for each word in the words set and accumulate them as a set