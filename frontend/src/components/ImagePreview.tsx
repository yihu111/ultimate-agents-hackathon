import { X } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useEffect } from 'react';

interface ImagePreviewProps {
  imageUrl?: string;
  variations?: string[];
  onClose: () => void;
}

export const ImagePreview = ({ imageUrl, variations, onClose }: ImagePreviewProps) => {
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };

    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, [onClose]);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-background/80 backdrop-blur-sm">
      <div className="relative max-h-[90vh] max-w-[90vw] animate-scale-in">
        <Button
          variant="outline"
          size="icon"
          onClick={onClose}
          className="absolute -top-12 right-0 bg-gradient-glass backdrop-blur-sm border-grid-border hover:bg-grid-hover"
        >
          <X className="h-4 w-4" />
        </Button>
        
        {variations && variations.length > 0 ? (
          <div className="grid grid-cols-2 gap-2 p-4 bg-background rounded-lg shadow-elevated">
            {variations.map((variationUrl, idx) => (
              <img
                key={idx}
                src={variationUrl}
                alt={`Variation ${idx + 1}`}
                className="max-h-[40vh] max-w-[40vw] object-contain rounded"
              />
            ))}
          </div>
        ) : imageUrl ? (
          <img
            src={imageUrl}
            alt="Preview"
            className="max-h-full max-w-full rounded-lg shadow-elevated"
          />
        ) : null}
      </div>
    </div>
  );
};