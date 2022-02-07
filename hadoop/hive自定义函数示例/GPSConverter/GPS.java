package GPSConverter;

/**
 * 坐标对象，由经纬度构成
 * Created by xfkang on 2018/3/28.
 */

public class GPS {
    private double lat;
    private double lon;

    public GPS(double lat, double lon) {
        this.lat = lat;
        this.lon = lon;
    }

    public double getLat() {
        return lat;
    }

    public void setLat(double lat) {
        this.lat = lat;
    }

    public double getLon() {
        return lon;
    }

    public void setLon(double lon) {
        this.lon = lon;
    }

    public String toString() {
        return "lat:" + lat + "," + "lon:" + lon;
    }
}
