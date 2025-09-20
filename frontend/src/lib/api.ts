const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

// Mock data for when backend is not available
const MOCK_IMAGES = [
  'https://images.unsplash.com/photo-1441986300917-64674bd600d8?w=400&h=400&fit=crop',
  'https://images.unsplash.com/photo-1560472354-b33ff0c44a43?w=400&h=400&fit=crop',
  'https://images.unsplash.com/photo-1526170375885-4d8ecf77b99f?w=400&h=400&fit=crop',
  'https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=400&h=400&fit=crop',
  'https://images.unsplash.com/photo-1583394838336-acd977736f90?w=400&h=400&fit=crop',
  'https://images.unsplash.com/photo-1525966222134-fcfa99b8ae77?w=400&h=400&fit=crop',
  'https://images.unsplash.com/photo-1572635196237-14b3f281503f?w=400&h=400&fit=crop',
  'https://images.unsplash.com/photo-1586297135537-94bc9ba060aa?w=400&h=400&fit=crop',
  'https://images.unsplash.com/photo-1593642632559-0c6d3fc62b89?w=400&h=400&fit=crop',
];

let useMockData = false;

export interface UploadResponse {
  prompt?: string | null;
  input_url?: string | null;
  ingestion_id?: string | null;
}

export interface ImageItem {
  slot: number;
  variant: number;
  url: string | null;
  kind: "base";
  status: "queued" | "running" | "done" | "error";
  version: number;
  updatedAt: number;
  meta?: Record<string, any>;
}

export interface GridResponse {
  status: "queued" | "running" | "done";
  rows: number;
  cols: number;
  progress: {
    done: number;
    total: number;
  };
  items: ImageItem[];
}

export interface SlotResponse {
  slot: number;
  items: ImageItem[];
}

export interface GenerateVariationsRequest {
  count: number;
}

export interface RegenerateRequest {
  slots: number[];
}

class ApiService {
  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    // If we're already using mock data, skip the API call
    if (useMockData) {
      throw new Error('Using mock data');
    }

    try {
      const url = `${API_BASE_URL}${endpoint}`;
      
      const response = await fetch(url, {
        ...options,
        headers: {
          ...options.headers,
        },
      });

      if (!response.ok) {
        throw new Error(`API Error: ${response.status} ${response.statusText}`);
      }

      return response.json();
    } catch (error) {
      // Switch to mock data if API is unavailable
      console.log('API unavailable, switching to mock data');
      useMockData = true;
      throw error;
    }
  }

  // File uploads removed; pass image_url directly to generateGrid

  async generateGrid(rows = 3, cols = 3, prompt?: string, imageUrl?: string): Promise<GridResponse> {
    try {
      const params = new URLSearchParams({ rows: String(rows), cols: String(cols) });
      if (prompt) params.set('prompt', prompt);
      if (imageUrl) params.set('image_url', imageUrl);
      return await this.request<GridResponse>(`/generate?${params.toString()}`, {
        method: 'POST',
      });
    } catch (error) {
      // Return mock response when API is unavailable
      console.log('Using mock grid generation');
      const mockItems: ImageItem[] = [];
      for (let i = 1; i <= rows * cols; i++) {
        mockItems.push({
          slot: i,
          variant: 0,
          url: MOCK_IMAGES[(i - 1) % MOCK_IMAGES.length],
          kind: "base",
          status: 'done',
          version: 1,
          updatedAt: Date.now(),
        });
      }
      
      return {
        status: 'done',
        rows,
        cols,
        progress: { done: rows * cols, total: rows * cols },
        items: mockItems,
      };
    }
  }

  async getGrid(): Promise<GridResponse> {
    try {
      return await this.request<GridResponse>('/grid');
    } catch (error) {
      // Return the same mock response as generateGrid
      return this.generateGrid(3, 3);
    }
  }

  async regenerateSlot(slot: number): Promise<ImageItem> {
    try {
      return await this.request<ImageItem>(`/slot/${slot}/generate`, {
        method: 'POST',
      });
    } catch (error) {
      console.log('Using mock slot regeneration');
      return {
        slot,
        variant: 0,
        url: MOCK_IMAGES[Math.floor(Math.random() * MOCK_IMAGES.length)],
        kind: "base",
        status: 'done',
        version: Date.now(),
        updatedAt: Date.now(),
      };
    }
  }

  async getSlot(slot: number): Promise<SlotResponse> {
    try {
      return await this.request<SlotResponse>(`/slot/${slot}`);
    } catch (error) {
      console.log('Using mock slot response');
      const mockItems: ImageItem[] = [
        {
          slot,
          variant: 0,
          url: MOCK_IMAGES[(slot - 1) % MOCK_IMAGES.length],
          kind: "base",
          status: 'done',
          version: 1,
          updatedAt: Date.now(),
        }
      ];
      
      for (let i = 1; i <= 3; i++) {
        mockItems.push({
          slot,
          variant: i,
          url: MOCK_IMAGES[Math.floor(Math.random() * MOCK_IMAGES.length)],
          kind: "base",
          status: 'done',
          version: 1,
          updatedAt: Date.now(),
        });
      }
      
      return {
        slot,
        items: mockItems,
      };
    }
  }

  async generateSlotVariations(slot: number, count = 3): Promise<SlotResponse> {
    try {
      return await this.request<SlotResponse>(`/slot/${slot}/variations/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ count }),
      });
    } catch (error) {
      console.log('Using mock variation generation');
      return this.getSlot(slot);
    }
  }

  async regenerateSlots(slots: number[]): Promise<GridResponse> {
    try {
      return await this.request<GridResponse>('/regenerate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ slots }),
      });
    } catch (error) {
      console.log('Using mock regeneration');
      const mockItems: ImageItem[] = [];
      for (let i = 1; i <= 9; i++) {
        mockItems.push({
          slot: i,
          variant: 0,
          url: slots.includes(i) 
            ? MOCK_IMAGES[Math.floor(Math.random() * MOCK_IMAGES.length)]
            : MOCK_IMAGES[(i - 1) % MOCK_IMAGES.length],
          kind: "base",
          status: 'done',
          version: slots.includes(i) ? Date.now() : 1,
          updatedAt: Date.now(),
        });
      }
      
      return {
        status: 'done',
        rows: 3,
        cols: 3,
        progress: { done: 9, total: 9 },
        items: mockItems,
      };
    }
  }
}

export const apiService = new ApiService();