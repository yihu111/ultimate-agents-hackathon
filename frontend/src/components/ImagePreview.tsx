import { X, Check } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useEffect } from 'react';
import { cn } from '@/lib/utils';

interface SelectedImage {
  gridItemId: string;
  imageUrl: string;
  variationIndex?: number;
}

interface ImagePreviewProps {
  imageUrl?: string;
  variations?: string[];
  onClose: () => void;
  selectedImages?: SelectedImage[];
  gridItemId?: string;
  onSelectVariation?: (variationIndex: number, imageUrl: string) => void;
  isVariationSelected?: (variationIndex: number) => boolean;
}

export const ImagePreview = ({ 
  imageUrl, 
  variations, 
  onClose, 
  selectedImages, 
  gridItemId, 
  onSelectVariation, 
  isVariationSelected 
}: ImagePreviewProps) => {
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
          <div className="grid grid-cols-2 gap-4 p-6 bg-background rounded-lg shadow-elevated">
            {variations.map((variationUrl, idx) => (
              <div
                key={idx}
                className={cn(
                  "relative group cursor-pointer rounded-lg overflow-hidden border-2 transition-all duration-300",
                  isVariationSelected?.(idx) ? "border-accent shadow-accent-glow" : "border-transparent hover:border-grid-border"
                )}
                onClick={() => onSelectVariation?.(idx, variationUrl)}
              >
                <img
                  src={variationUrl}
                  alt={`Variation ${idx + 1}`}
                  className="max-h-[40vh] max-w-[40vw] w-full object-contain"
                />
                
                {/* Selection indicator */}
                {isVariationSelected?.(idx) && (
                  <div className="absolute top-3 left-3 rounded-full bg-accent p-2">
                    <Check className="h-4 w-4 text-accent-foreground" />
                  </div>
                )}
                
                {/* Hover overlay */}
                <div className="absolute inset-0 bg-black/20 opacity-0 group-hover:opacity-100 transition-opacity duration-200" />
                
                {/* Click hint */}
                <div className="absolute bottom-3 right-3 px-2 py-1 bg-background/80 rounded text-xs text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity duration-200">
                  Click to select
                </div>
              </div>
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