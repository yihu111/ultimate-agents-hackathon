import { useState, useEffect, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { GridCell } from '@/components/GridCell';
import { ImagePreview } from '@/components/ImagePreview';
import { Toolbar } from '@/components/Toolbar';
import { Skeleton } from '@/components/ui/skeleton';
import { apiService, type GridResponse, type ImageItem } from '@/lib/api';
import { toast } from 'sonner';
import { ArrowLeft, Save } from 'lucide-react';

interface AdGridProps {
  uploadedImage: File;
  uploadResponse: any;
  onBack: () => void;
  initialPrompt?: string;
}

interface SelectedImage {
  slot: number;
  imageUrl: string;
  variationIndex?: number;
}

const AdGrid = ({ uploadedImage, uploadResponse, onBack, initialPrompt }: AdGridProps) => {
  const [gridItems, setGridItems] = useState<ImageItem[]>([]);
  const [selectedImages, setSelectedImages] = useState<SelectedImage[]>([]);
  const [gridData, setGridData] = useState<GridResponse | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [zoomedImage, setZoomedImage] = useState<{ imageUrl: string; variations?: string[] } | null>(null);
  const [adPrompt, setAdPrompt] = useState(initialPrompt || '');
  const [isSavingPrompt, setIsSavingPrompt] = useState(false);
  const [isRegenerating, setIsRegenerating] = useState<Record<number, boolean>>({});
  const [isGeneratingVariations, setIsGeneratingVariations] = useState<Record<number, boolean>>({});

  useEffect(() => {
    startGeneration();
  }, []);

  const startGeneration = async () => {
    try {
      setIsGenerating(true);
      
      // Initialize empty grid items with loading state
      const initialItems: ImageItem[] = [];
      for (let i = 1; i <= 9; i++) {
        initialItems.push({
          slot: i,
          variant: 0,
          url: '',
          status: 'queued',
          version: 1,
          updatedAt: new Date().toISOString(),
        });
      }
      setGridItems(initialItems);
      
      // Start generation
      const response = await apiService.generateGrid(3, 3);
      setGridData(response);
      
      // If generation is complete immediately, update grid
      if (response.status === 'done') {
        setGridItems(response.items);
        setIsGenerating(false);
      } else {
        // Start polling
        pollGrid();
      }
    } catch (error) {
      console.error('Generation failed:', error);
      toast.error('Failed to generate images');
      setIsGenerating(false);
    }
  };

  const pollGrid = useCallback(async () => {
    try {
      const response = await apiService.getGrid();
      setGridData(response);
      setGridItems(response.items);
      
      if (response.status === 'done') {
        setIsGenerating(false);
        return;
      }
      
      // Continue polling every 1-2 seconds
      setTimeout(pollGrid, 1500);
    } catch (error) {
      console.error('Polling failed:', error);
      setIsGenerating(false);
    }
  }, []);

  const handleImageSelect = (slot: number, imageUrl: string, variationIndex?: number) => {
    const newSelection: SelectedImage = {
      slot,
      imageUrl,
      variationIndex,
    };
    
    const existingIndex = selectedImages.findIndex(img => 
      img.slot === slot && img.variationIndex === variationIndex
    );
    
    if (existingIndex >= 0) {
      // Remove selection
      const newSelectedImages = [...selectedImages];
      newSelectedImages.splice(existingIndex, 1);
      setSelectedImages(newSelectedImages);
    } else {
      // Add selection
      const newSelectedImages = [...selectedImages, newSelection];
      setSelectedImages(newSelectedImages);
    }
  };

  const isImageSelected = (slot: number, variationIndex?: number): boolean => {
    return selectedImages.some(img => 
      img.slot === slot && img.variationIndex === variationIndex
    );
  };

  const handleItemRegenerate = async (slot: number) => {
    try {
      setIsRegenerating({ ...isRegenerating, [slot]: true });
      
      const newItem = await apiService.regenerateSlot(slot);
      
      // Update the specific item in the grid
      setGridItems(prevItems => 
        prevItems.map(item => 
          item.slot === slot ? newItem : item
        )
      );
      
      toast.success('Image regenerated successfully');
    } catch (error) {
      console.error('Regeneration failed:', error);
      toast.error('Failed to regenerate image');
    } finally {
      setIsRegenerating({ ...isRegenerating, [slot]: false });
    }
  };

  const handleItemZoom = (imageUrl: string, variations?: string[]) => {
    setZoomedImage({ imageUrl, variations });
  };

  const handleGenerateVariations = async (slot: number) => {
    try {
      setIsGeneratingVariations({ ...isGeneratingVariations, [slot]: true });
      
      // First check if variations already exist
      const slotData = await apiService.getSlot(slot);
      
      if (slotData.items.length > 1) {
        // Variations already exist, show them
        const baseItem = slotData.items.find(item => item.variant === 0);
        const allVariations = slotData.items.map(item => item.url);
        
        if (baseItem) {
          setZoomedImage({ 
            imageUrl: baseItem.url, 
            variations: allVariations 
          });
        }
      } else {
        // Generate new variations
        const response = await apiService.generateSlotVariations(slot, 3);
        
        // Show all variations (base + new ones) in 2x2 grid
        const allVariations = response.items.map(item => item.url);
        const baseItem = response.items.find(item => item.variant === 0);
        
        if (baseItem) {
          setZoomedImage({ 
            imageUrl: baseItem.url, 
            variations: allVariations 
          });
        }
        
        toast.success('Variations generated successfully');
      }
    } catch (error) {
      console.error('Variation generation failed:', error);
      toast.error('Failed to generate variations');
    } finally {
      setIsGeneratingVariations({ ...isGeneratingVariations, [slot]: false });
    }
  };

  const handleRegenerateAll = async () => {
    try {
      setIsGenerating(true);
      
      if (selectedImages.length > 0) {
        // Regenerate only selected images
        const slotsToRegenerate = [...new Set(selectedImages.map(img => img.slot))];
        const response = await apiService.regenerateSlots(slotsToRegenerate);
        setGridItems(response.items);
        
        // Clear selections
        setSelectedImages([]);
        
        toast.success(`Regenerated ${slotsToRegenerate.length} selected images`);
      } else {
        // Regenerate all images
        const allSlots = Array.from({ length: 9 }, (_, i) => i + 1);
        const response = await apiService.regenerateSlots(allSlots);
        setGridItems(response.items);
        toast.success('Regenerated all images');
      }
      
    } catch (error) {
      console.error('Regeneration failed:', error);
      toast.error('Failed to regenerate images');
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="min-h-screen bg-background p-6">
      <div className="mx-auto max-w-7xl">
        {/* Header */}
        <div className="mb-8">
          <div>
            <h1 className="text-3xl font-bold">Generated Ad Variations</h1>
            <p className="text-muted-foreground">
              Your ad variations are ready. Select the ones you like best.
            </p>
          </div>
        </div>

        {/* Prompt Section */}
        {adPrompt && (
          <div className="mb-6 rounded-lg border bg-card p-4">
            <div className="flex items-center justify-between mb-2">
              <label className="text-sm font-medium">Ad Prompt</label>
            </div>
            <div className="flex gap-2">
              <Input
                value={adPrompt}
                onChange={(e) => setAdPrompt(e.target.value)}
                placeholder="Describe your product, target audience, or specific requirements..."
                className="flex-1"
              />
              <Button 
                variant="outline" 
                size="sm" 
                disabled={isSavingPrompt}
                className="flex items-center gap-2"
              >
                <Save className="h-4 w-4" />
                {isSavingPrompt ? 'Saving...' : 'Save'}
              </Button>
            </div>
          </div>
        )}

        {/* Status */}
        {isGenerating && (
          <div className="mb-6 text-center">
            <div className="flex items-center justify-center gap-2 text-muted-foreground">
              <div className="h-4 w-4 animate-spin rounded-full border-2 border-primary border-t-transparent" />
              Generating your ad variations... ({gridData?.progress.done || 0}/{gridData?.progress.total || 9})
            </div>
          </div>
        )}

        {/* Toolbar */}
        <Toolbar
          totalGenerated={gridData?.progress.done || 0}
          selectedCount={selectedImages.length}
          selectedImages={selectedImages.map(img => img.imageUrl)}
          onRegenerateAll={handleRegenerateAll}
          onBack={onBack}
          onReset={startGeneration}
        />

        {/* Grid */}
        <div className="mt-8">
          <div className="grid grid-cols-3 gap-4 max-w-4xl mx-auto">
            {gridItems.map((item) => (
              <GridCell
                key={`${item.slot}-${item.version}-${item.updatedAt}`}
                imageUrl={item.status === 'done' ? item.url : null}
                isLoading={item.status !== 'done'}
                isSelected={isImageSelected(item.slot)}
                onSelect={() => item.status === 'done' && handleImageSelect(item.slot, item.url)}
                onRegenerate={() => handleItemRegenerate(item.slot)}
                onZoom={() => item.status === 'done' && handleItemZoom(item.url)}
                onGenerateVariations={() => handleGenerateVariations(item.slot)}
                index={item.slot - 1}
                onSelectVariation={(variationIndex, imageUrl) => 
                  handleImageSelect(item.slot, imageUrl, variationIndex)
                }
                isVariationSelected={(variationIndex) => 
                  isImageSelected(item.slot, variationIndex)
                }
              />
            ))}
          </div>
        </div>
      </div>

      {/* Image Preview Modal */}
      {zoomedImage && (
        <ImagePreview
          imageUrl={zoomedImage.imageUrl}
          variations={zoomedImage.variations}
          onClose={() => setZoomedImage(null)}
          selectedImages={selectedImages.map(img => ({
            gridItemId: img.slot.toString(),
            imageUrl: img.imageUrl,
            variationIndex: img.variationIndex
          }))}
          gridItemId={gridItems.find(item => item.url === zoomedImage.imageUrl)?.slot.toString()}
          onSelectVariation={(variationIndex, imageUrl) => {
            const gridItem = gridItems.find(item => item.url === zoomedImage.imageUrl);
            if (gridItem) {
              handleImageSelect(gridItem.slot, imageUrl, variationIndex);
            }
          }}
          isVariationSelected={(variationIndex) => {
            const gridItem = gridItems.find(item => item.url === zoomedImage.imageUrl);
            return gridItem ? isImageSelected(gridItem.slot, variationIndex) : false;
          }}
        />
      )}
    </div>
  );
};

export { AdGrid };