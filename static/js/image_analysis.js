class OrchidAnalyzer {
    async analyzeImage(imageElement) {
        const imageUrl = imageElement.currentSrc || imageElement.src;
        if (!imageUrl) {
            throw new Error('The selected image could not be read.');
        }

        const imageResponse = await fetch(imageUrl);
        if (!imageResponse.ok) {
            throw new Error('The selected image could not be prepared for analysis.');
        }

        const formData = new FormData();
        formData.append('file', await imageResponse.blob(), 'orchid-image.jpg');
        const response = await fetch('/api/analyze', {
            method: 'POST',
            body: formData
        });
        const result = await response.json();
        if (!response.ok || !result.success) {
            throw new Error(result.error || `Image analysis failed (${response.status}).`);
        }
        return result;
    }
}

const orchidAnalyzer = new OrchidAnalyzer();
