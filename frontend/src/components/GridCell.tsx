import { useState } from 'react';
import { Check, RotateCcw, ZoomIn, Loader2, Grid3X3 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

interface GridCellProps {
  imageUrl?: string;
  isLoading?: boolean;
  isSelected?: boolean;
  onSelect?: () => void;
  onRegenerate?: () => void;
  onZoom?: (imageUrl?: string, variations?: string[]) => void;
  onGenerateVariations?: () => void;
  variations?: string[];
  isGeneratingVariations?: boolean;
  index: number;
}

export const GridCell = ({
  imageUrl,
  isLoading,
  isSelected,
  onSelect,
  onRegenerate,
  onZoom,
  onGenerateVariations,
  variations,
  isGeneratingVariations,
  index
}: GridCellProps) => {
  const [isHovered, setIsHovered] = useState(false);
  const [imageLoaded, setImageLoaded] = useState(false);

  return (
    <div
      className={cn(
        "relative aspect-square rounded-lg border border-grid-border bg-grid-bg transition-all duration-300 overflow-hidden group",
        isHovered && "border-primary shadow-glow",
        isSelected && "border-accent shadow-accent-glow",
        "animate-grid-item-in"
      )}
      style={{ animationDelay: `${index * 0.05}s` }}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {/* Loading State */}
      {(isLoading || isGeneratingVariations) && (
        <div className="absolute inset-0 flex items-center justify-center bg-grid-bg z-10">
          <div className="flex flex-col items-center space-y-3">
            <Loader2 className="h-8 w-8 animate-spin text-primary" />
            <span className="text-sm text-muted-foreground">
              {isGeneratingVariations ? 'Creating variations...' : 'Generating...'}
            </span>
          </div>
        </div>
      )}

      {/* Variations Grid (2x2) */}
      {variations && variations.length > 0 && (
        <div className="grid grid-cols-2 gap-0.5 h-full w-full">
          {variations.map((variationUrl, idx) => (
            <img
              key={idx}
              src={variationUrl}
              alt={`Variation ${idx + 1}`}
              className={cn(
                "h-full w-full object-cover transition-all duration-300",
                isHovered && "scale-105"
              )}
              onLoad={() => setImageLoaded(true)}
            />
          ))}
        </div>
      )}

      {/* Single Image */}
      {imageUrl && !variations && (
        <img
          src={imageUrl}
          alt={`Ad variation ${index + 1}`}
          className={cn(
            "h-full w-full object-cover transition-all duration-300",
            !imageLoaded && "opacity-0",
            imageLoaded && "opacity-100",
            isHovered && "scale-105"
          )}
          onLoad={() => setImageLoaded(true)}
        />
      )}

      {/* Placeholder */}
      {!imageUrl && !isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-grid-bg">
          <div className="text-center">
            <div className="h-12 w-12 mx-auto mb-2 rounded-lg bg-muted flex items-center justify-center">
              <span className="text-muted-foreground font-mono text-sm">
                {index + 1}
              </span>
            </div>
            <span className="text-xs text-muted-foreground">Empty</span>
          </div>
        </div>
      )}

      {/* Selection Indicator */}
      {isSelected && (
        <div className="absolute top-2 left-2 rounded-full bg-accent p-1">
          <Check className="h-4 w-4 text-accent-foreground" />
        </div>
      )}

      {/* Hover Actions */}
      {(isHovered || isSelected) && imageUrl && !isLoading && !isGeneratingVariations && (
        <div className="absolute inset-0 bg-background/20 backdrop-blur-sm">
          <div className="absolute bottom-2 left-2 right-2 flex space-x-1">
            <Button
              size="sm"
              variant={isSelected ? "default" : "secondary"}
              onClick={onSelect}
              className="flex-1 bg-gradient-glass backdrop-blur-sm"
            >
              <Check className="h-4 w-4" />
            </Button>
            
            {!variations && (
              <Button
                size="sm"
                variant="secondary"
                onClick={onGenerateVariations}
                className="bg-gradient-glass backdrop-blur-sm"
                title="Generate Variations"
              >
                <Grid3X3 className="h-4 w-4" />
              </Button>
            )}
            
            <Button
              size="sm"
              variant="secondary"
              onClick={onRegenerate}
              className="bg-gradient-glass backdrop-blur-sm"
            >
              <RotateCcw className="h-4 w-4" />
            </Button>
            
            <Button
              size="sm"
              variant="secondary"
              onClick={() => onZoom?.(imageUrl, variations)}
              className="bg-gradient-glass backdrop-blur-sm"
            >
              <ZoomIn className="h-4 w-4" />
            </Button>
          </div>
        </div>
      )}
    </div>
  );
};