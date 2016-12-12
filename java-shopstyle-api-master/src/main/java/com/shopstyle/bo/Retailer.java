package com.shopstyle.bo;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

@JsonIgnoreProperties(ignoreUnknown = true)
public class Retailer implements SearchFilter
{
    private long id;
    private String name;

    @JsonCreator
    public Retailer(@JsonProperty("id") long id, @JsonProperty("name") String name)
    {
        this.id = id;
        this.name = name;
    }
    
    @Override
    public String getFilterId()
    {
        return "r" + getId();
    }
  
    /**
     * Returns the unique identifier of this retailer
     */
    public long getId()
    {
        return id;
    }

    /**
     * Returns the display name of this retailer
     */
    public String getName()
    {
        return name;
    }
    public void setId(long i)
    {
        id = i;
    }
    public void setName(String n)
    {
        name = n;
    }
    public String toString()
    {
        return "Retailer{" + "id=" + id + ", name=" + name + "}";
    }
}
