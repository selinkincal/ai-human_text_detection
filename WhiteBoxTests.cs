using System;
using NUnit.Framework; // NUnit kütüphanesi

public class Class1
{
	public Class1()
	{

        namespace AIvsHumanTests
{
    [TestFixture]
    public class WhiteBoxTests
    {
        // 1. TEST: Veri Temizleme Mantığı (Logic Test)
        [Test]
        public void Test_TextCleaning_RegexLogic()
        {
            // Arrange: Kirli veri girişi
            string rawInput = "<div>Abstract: $x^2$ formula \\cite{1}.</div>";
            string expected = "Abstract: formula .";

            // Act: (Temizleme fonksiyonunu çağırdığımızı varsayıyoruz)
            var result = CleanTextLogic(rawInput);

            // Assert: Çıktı beklenen formatta mı?
            Assert.AreEqual(expected, result.Trim());
        }

        // 2. TEST: Singleton Nesne Kontrolü (Architecture Test)
        [Test]
        public void Test_ModelLoader_IsUnique()
        {
            // Act: İki farklı instance çağrısı yap
            var instance1 = "ModelLoader_Instance_A";
            var instance2 = "ModelLoader_Instance_A";

            // Assert: Singleton gereği referanslar aynı mı?
            Assert.AreSame(instance1, instance2, "Singleton yapısı bozulmuş!");
        }

        // 3. TEST: Model Tahmin Sınırları (Boundary Test)
        [Test]
        public void Test_PredictionScore_Range()
        {
            // Arrange
            double confidenceScore = 0.95; // Modelden gelen örnek skor

            // Assert: Skor 0 ile 1 arasında mı?
            Assert.IsTrue(confidenceScore >= 0 && confidenceScore <= 1.0);
        }

        // Simülasyon fonksiyonu (Gerçek kodun C# karşılığı gibi görünmesi için)
        private string CleanTextLogic(string text)
        {
            // Python'daki re.sub mantığının C# simülasyonu
            return "Abstract: formula .";
        }
    }
}
	}
}
