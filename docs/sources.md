# Sources for the State of the Art section (checked 2026-11-23)

Each entry below was opened and checked; the notes come from the abstract or the publication page.

1. Rosen, S. (1974). Hedonic Prices and Implicit Markets: Product Differentiation in Pure Competition.
   Journal of Political Economy, 82(1), 34-55. https://doi.org/10.1086/260169
   - The theory behind hedonic pricing: a property's price is the sum of implicit prices of its
     characteristics (size, rooms, location). Hedonic regression (linear regression on characteristics)
     is the classic appraisal method and is what our Linear Regression model follows.

2. จิรพล สังข์โพธิ์ และ ศราวุธ แรมจันทร์ (2563). การพยากรณ์ราคาคอนโดมิเนียมในกรุงเทพมหานครโดยเทคนิค Web Mining
   (Predicting Condominium Price in Bangkok Using Web Mining Techniques). วารสารวิจัยและพัฒนา มศว,
   ปีที่ 12 ฉบับที่ 24 (ก.ค.-ธ.ค. 2563). https://ejournals.swu.ac.th/index.php/swurd/article/view/13524
   - Web-crawled data of 1,465 Bangkok condo projects from hipflat.co.th, 15 attributes.
   - Compared 5 supervised models; deep learning was best, relative error 17.53%.
   - Target is price per square metre at project level (not unit level), so not directly comparable to us.
   - Key drivers found: number of units in the project, distance to BTS, building age.

3. Yazdani, M. (2021). Machine Learning, Deep Learning, and Hedonic Methods for Real Estate Price Prediction.
   arXiv:2110.07151. https://arxiv.org/abs/2110.07151
   - Compares hedonic regression with random forest and neural networks on house sales;
     concludes random forest and neural networks can be better alternatives to hedonic regression.

4. Gorjian, M. (2025). Integrating Machine Learning and Hedonic Regression for Housing Price Prediction:
   A Systematic International Review of Model Performance and Interpretability. MPRA Paper 125676.
   https://econpapers.repec.org/RePEc:pra:mprapa:125676
   - Review of studies: hedonic models keep strong explanatory power (interpretable coefficients),
     while tree ensembles (random forest, gradient boosting) and neural networks usually predict more precisely.

How these map to our comparison:
- Classic method (hedonic / linear regression): our main Linear Regression model.
- Current practice in research (tree ensembles): our Gradient Boosting reference model on the same data.
- Simple market rule (price per sqm of the area x size): our baseline.
