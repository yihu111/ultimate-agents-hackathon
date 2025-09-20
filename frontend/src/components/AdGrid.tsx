import { useState, useEffect, useCallback } from 'react';
import { GridCell } from './GridCell';
import { Toolbar } from './Toolbar';
import { ImagePreview } from './ImagePreview';
import { apiService, GenerateResponse } from '@/lib/api';
import { useToast } from '@/hooks/use-toast';

interface AdGridProps {
  uploadedImage: File;
  uploadResponse: any;
  onBack: () => void;
}

interface GridItem {
  id: string;
  imageUrl?: string;
  isLoading: boolean;
  index: number;
  variations?: string[];
  isGeneratingVariations?: boolean;
}

export const AdGrid = ({ uploadedImage, uploadResponse, onBack }: AdGridProps) => {
  const [gridItems, setGridItems] = useState<GridItem[]>([]);
  const [selectedItems, setSelectedItems] = useState<Set<string>>(new Set());
  const [previewImage, setPreviewImage] = useState<string | null>(null);
  const [previewVariations, setPreviewVariations] = useState<string[] | undefined>(undefined);
  const [isGenerating, setIsGenerating] = useState(false);
  const [gridData, setGridData] = useState<GenerateResponse | null>(null);
  const { toast } = useToast();

  // Initialize grid and start generation
  useEffect(() => {
    const initialItems: GridItem[] = Array.from({ length: 9 }, (_, index) => ({
      id: `item-${index}`,
      imageUrl: undefined,
      isLoading: true,
      index: index + 1, // 1-indexed for API
    }));
    setGridItems(initialItems);
    startGeneration();
  }, [uploadedImage]);

  const startGeneration = async () => {
    try {
      setIsGenerating(true);
      // Start generation process
      await apiService.generateGrid(3, 3);
      // Start polling for updates
      pollGrid();
    } catch (error) {
      console.error('Generation failed:', error);
      toast({
        title: "Generation Failed",
        description: "Failed to generate variations. Please try again.",
        variant: "destructive",
      });
    }
  };

  const pollGrid = useCallback(async () => {
    try {
      const response = await apiService.getGrid(3, 3);
      setGridData(response);
      
      // Update grid items with new images
      setGridItems(prev => 
        prev.map(item => ({
          ...item,
          imageUrl: response.images[item.index.toString()] || undefined,
          isLoading: !response.images[item.index.toString()],
        }))
      );

      setIsGenerating(response.status !== 'done');
      
      // Continue polling if not done
      if (response.status !== 'done') {
        setTimeout(pollGrid, 1500);
      }
    } catch (error) {
      console.error('Polling failed:', error);
      setIsGenerating(false);
    }
  }, []);

  const handleItemSelect = (itemId: string) => {
    setSelectedItems(prev => {
      const newSet = new Set(prev);
      if (newSet.has(itemId)) {
        newSet.delete(itemId);
      } else {
        newSet.add(itemId);
      }
      return newSet;
    });
  };

  const handleItemRegenerate = async (itemId: string) => {
    const item = gridItems.find(item => item.id === itemId);
    if (!item) return;

    // Set loading state for this item
    setGridItems(prev =>
      prev.map(prevItem =>
        prevItem.id === itemId
          ? { ...prevItem, isLoading: true, imageUrl: undefined }
          : prevItem
      )
    );

    try {
      // Regenerate this specific slot
      const response = await apiService.regenerateSlots([item.index]);
      
      // Update the grid with the new image
      setGridItems(prev =>
        prev.map(prevItem =>
          prevItem.id === itemId
            ? { ...prevItem, isLoading: false, imageUrl: response.images[item.index.toString()] }
            : prevItem
        )
      );
    } catch (error) {
      console.error('Regeneration failed:', error);
      setGridItems(prev =>
        prev.map(prevItem =>
          prevItem.id === itemId
            ? { ...prevItem, isLoading: false }
            : prevItem
        )
      );
      toast({
        title: "Regeneration Failed",
        description: "Failed to regenerate image. Please try again.",
        variant: "destructive",
      });
    }
  };

  const handleItemZoom = (imageUrl?: string, variations?: string[]) => {
    setPreviewImage(imageUrl || null);
    setPreviewVariations(variations);
  };

  const handleGenerateVariations = async (itemId: string) => {
    const item = gridItems.find(item => item.id === itemId);
    if (!item) return;

    // Set generating variations state
    setGridItems(prev =>
      prev.map(prevItem =>
        prevItem.id === itemId
          ? { ...prevItem, isGeneratingVariations: true }
          : prevItem
      )
    );

    try {
      const response = await apiService.generateVariations(item.index, 4);
      
      // Build 2x2 grid: [base, v1, v2, v3] where base is the original image
      const variationsGrid = [item.imageUrl!];
      if (response.urls.length >= 3) {
        variationsGrid.push(...response.urls.slice(0, 3));
      } else {
        // Fill remaining slots with available variations
        variationsGrid.push(...response.urls);
        while (variationsGrid.length < 4) {
          variationsGrid.push(item.imageUrl!); // Fallback to base image
        }
      }
      
      // Update the grid with variations (original + 3 new)
      setGridItems(prev =>
        prev.map(prevItem =>
          prevItem.id === itemId
            ? { 
                ...prevItem, 
                isGeneratingVariations: false, 
                variations: variationsGrid
              }
            : prevItem
        )
      );
    } catch (error) {
      console.error('Variation generation failed:', error);
      setGridItems(prev =>
        prev.map(prevItem =>
          prevItem.id === itemId
            ? { ...prevItem, isGeneratingVariations: false }
            : prevItem
        )
      );
      toast({
        title: "Variation Generation Failed",
        description: "Failed to generate variations. Please try again.",
        variant: "destructive",
      });
    }
  };

  const handleRegenerateAll = async () => {
    const itemsToRegenerate = selectedItems.size > 0 ? 
      Array.from(selectedItems) : 
      gridItems.map(item => item.id);

    const slotsToRegenerate = itemsToRegenerate.map(itemId => {
      const item = gridItems.find(item => item.id === itemId);
      return item?.index;
    }).filter(Boolean) as number[];

    // Set loading state for items to be regenerated
    setGridItems(prev =>
      prev.map(item => 
        itemsToRegenerate.includes(item.id)
          ? { ...item, isLoading: true, imageUrl: undefined }
          : item
      )
    );

    // Clear selection if we regenerated selected items
    if (selectedItems.size > 0) {
      setSelectedItems(new Set());
    }

    try {
      const response = await apiService.regenerateSlots(slotsToRegenerate);
      
      // Update grid with new images
      setGridItems(prev =>
        prev.map(item => ({
          ...item,
          imageUrl: response.images[item.index.toString()] || item.imageUrl,
          isLoading: itemsToRegenerate.includes(item.id) ? false : item.isLoading,
        }))
      );
    } catch (error) {
      console.error('Regeneration failed:', error);
      // Reset loading states on error
      setGridItems(prev =>
        prev.map(item => 
          itemsToRegenerate.includes(item.id)
            ? { ...item, isLoading: false }
            : item
        )
      );
      toast({
        title: "Regeneration Failed",
        description: "Failed to regenerate images. Please try again.",
        variant: "destructive",
      });
    }
  };

  const totalGenerated = gridItems.filter(item => item.imageUrl && !item.isLoading).length;

  return (
    <div className="min-h-screen bg-background p-6">
      <div className="mx-auto max-w-7xl">
        {/* Header */}
        <div className="mb-8 flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold bg-gradient-primary bg-clip-text text-transparent">
              Generated Variations
            </h1>
            <p className="text-muted-foreground">
              Your ad variations are ready. Select the ones you like best.
            </p>
          </div>
        </div>

        {/* Toolbar */}
        <Toolbar
          totalGenerated={totalGenerated}
          selectedCount={selectedItems.size}
          onRegenerateAll={handleRegenerateAll}
          onBack={onBack}
          onReset={() => apiService.reset()}
        />

        {/* Grid with Axis Labels */}
        <div className="relative">
          {/* Y-axis label */}
          {gridData?.y_axis_label && (
            <div className="absolute -left-16 top-1/2 -translate-y-1/2 -rotate-90 text-sm font-medium text-muted-foreground whitespace-nowrap">
              {gridData.y_axis_label}
            </div>
          )}
          
          {/* X-axis label */}
          {gridData?.x_axis_label && (
            <div className="text-center mb-4">
              <span className="text-sm font-medium text-muted-foreground">
                {gridData.x_axis_label}
              </span>
            </div>
          )}
          
          <div className="grid grid-cols-3 gap-4">
          {gridItems.map((item, index) => (
            <GridCell
              key={item.id}
              imageUrl={item.imageUrl}
              isLoading={item.isLoading}
              isSelected={selectedItems.has(item.id)}
              onSelect={() => handleItemSelect(item.id)}
              onRegenerate={() => handleItemRegenerate(item.id)}
              onZoom={() => handleItemZoom(item.imageUrl, item.variations)}
              onGenerateVariations={() => handleGenerateVariations(item.id)}
              variations={item.variations}
              isGeneratingVariations={item.isGeneratingVariations}
              index={index}
            />
          ))}
          </div>
        </div>
      </div>

      {/* Image Preview Modal */}
      {(previewImage || previewVariations) && (
        <ImagePreview
          imageUrl={previewImage || undefined}
          variations={previewVariations}
          onClose={() => {
            setPreviewImage(null);
            setPreviewVariations(undefined);
          }}
        />
      )}
    </div>
  );
};