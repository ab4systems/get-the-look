package com.shopstyle.bo;

/**
 * Wrapper around a specific image URL, providing metadata about the actual image. For a given
 * product, the same picture will be available in different {@link #getSizes() sizes}.
 */
public class Image
{
    private String id;
    private int width;
    private int height;
    private String url;

    public int getWidth()
    {
        return width;
    }

    public void setWidth(int width)
    {
        this.width = width;
    }

    public int getHeight()
    {
        return height;
    }

    public void setHeight(int height)
    {
        this.height = height;
    }

    /**
     * Returns the URL of the image file
     */
    public String getUrl()
    {
        return url;
    }

    public void setUrl(String url)
    {
        this.url = url;
    }

    /**
     * Returns the unique identifier of an image
     */
    public String getId()
    {
        return id;
    }

    public void setId(String id)
    {
        this.id = id;
    }
    public String toString() 
    {
        return "Image{id=" + id + ", width=" + String.valueOf(width) + ", height="
                + String.valueOf(height) + ", url=" + url + "}";
    }
}
