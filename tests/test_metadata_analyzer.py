import unittest
from src.metadata_analyzer import detect_ai_signature

class TestMetadataAnalyzer(unittest.TestCase):
    
    def test_detect_ai_signature_with_ai_tool(self):
        # AI yazılım izi içeren sahte (mock) EXIF verisi
        mock_tags = {
            "Image Software": "Midjourney 5.2",
            "Image Make": "Unknown",
            "EXIF ExifImageWidth": "1024"
        }
        
        is_ai, signatures = detect_ai_signature(mock_tags)
        
        self.assertTrue(is_ai, "AI izi bulunmalıydı fakat bulunamadı.")
        self.assertEqual(len(signatures), 1)
        self.assertIn("midjourney", signatures[0].lower())
        
    def test_detect_ai_signature_clean(self):
        # Temiz (AI izi içermeyen) sahte EXIF verisi
        mock_tags = {
            "Image Software": "Adobe Photoshop 2023",
            "Image Make": "Canon",
            "Image Model": "Canon EOS R5"
        }
        
        is_ai, signatures = detect_ai_signature(mock_tags)
        
        self.assertFalse(is_ai, "Temiz veride AI izi bulundu uyarısı verdi.")
        self.assertEqual(len(signatures), 0)

if __name__ == "__main__":
    unittest.main()
