# BLS MLflow PostgreSQL Setup Documentation
*Remy Stewart, BLS Civic Digital Fellow, Summer 2022. The following workflow was established by BLS Data Scientist Lowell Mason.*

The remote BLS server is supported by a PostgreSQL database that provides a designated location for parameter and metric storage for created MLflow experiments. This document delineates how to establish a PostgreSQL container within the Red Hat Enterprise Linux 8 (RHEL8) Server that is currently hosting the remote BLS server.

Hosting the PostgreSQL server as a container image rather than installing the entire database locally streamlines set-up and promotes replicability for other prospective MLflow servers across the BLS. The following instructions configures the PostgreSQL container to store data via a locally mounted directory within the server.

All of the following commands are ran via the terminal.

1. [Podman](https://github.com/containers/podman) hosts and maintains containers within Linux systems such as the RHEL8 server. We therefore first install the RHEL8 Podman distribution via the `container-tools` module as follows:
    ```bash
    $ sudo yum module enable -y container-tools:rhel8
    
    $ sudo yum module install -y container-tools:rhel8
     ```
2. Log into Podman and then download the PostgreSQL container images hosted within Redhat’s container catalogue.
    ```bash
    $ podman login -u="<your_username>" -p="<your_token>" registry.redhat.io
    
    $ podman pull registry.redhat.io/rhel8/postgresql-12:1-109.1655143367 
     ```


3. Let’s inspect some of the parameters associated with the container image.

    - The following command determines the container images’ embedded user, which will need to match the owner of the local directory that will be mounted to the PostgreSQL container to locally preserve logged MLflow data:
        ```bash
        $ sudo podman inspect registry.redhat.io/rhel8/postgresql-12 | grep User
        "User": "26"
        ```
    - Containers employ volumes for their internal data storage, in which the following prints out the container image's embedded volume path that we’ll use when establishing the local directory mount:
        ```bash
        $ sudo podman inspect registry.redhat.io/rhel8/postgresql-12 | grep HOME 
        "HOME": "<Embedded Path>"
        ```
    - This final inspection delineates the basic usage for the image, outlining the command syntax we will build from to initialize the PostgreSQL container.  
        ```bash 
        $ sudo podman inspect registry.redhat.io/rhel8/postgresql-12 | grep usage "usage": "podman run -d --name postgresql_database -e POSTGRESQL_USER=user -e  POSTGRESQL_PASSWORD=pass -e POSTGRESQL_DATABASE=db -p <####>:<####> rhel8/postgresql-12"
        ```
      

4. We’ll then create the local directory that will be mounted to the container:

    ```bash
    $ sudo mkdir **<Local Directory>**

    $ sudo chown 26:26 **<Local Directory>**

    $ sudo setfacl -m u:26:-wx **<Local Directory>**
    ```

5. We can proceed with creating the container itself via `podman run`. Note that the passed parameters include a set name for the PostgreSQL container, designating the username and password for the database, a chosen database name, and the newly created directory.

```bash
$ sudo podman run --name **<PostgreSQL Container Name>** \
                 -e POSTGRESQL_USER=**<DB User>** \
                 -e POSTGRESQL_PASSWORD=**<DB Password>** \
                 -e POSTGRESQL_DATABASE=**<DB Name>** \
                 -p <####>:<####> \
                 -v **<Local Directory>**:**<Embedded Path>** \
                 --net host \
                 rhel8/postgresql-12
```
6. We’ll then want to establish a systemd unit file used to manage the container.

```bash
$ sudo touch /etc/system/system/**<PostgreSQL Container Name>.**service
```

7.  Systemd files are configuration files that can delineate software behavior such as the following file contents designed to restart the database following a rebooting of the RHEL8 server. Said files designate daemon programs that run in the background without the user needing to interact with the program directly. These instructions can be added to the file via a text editor such as vi or nano.
```bash
[Unit]
Description=PostgreSQL Podman Container
After=network.target

[Service]
Type=simple
TimeoutStartSec=5m
ExecStartPre=-/usr/bin/podman rm "**<PostgreSQL Container Name>**"
ExecStart=/usr/bin/podman run --name **<PostgreSQL Container Name>** -e POSTGRESQL_USER=**<DB User>** -e POSTGRESQL_PASSWORD=**<DB Password>** -e POSTGRESQL_DATABASE=**<DB Name>** -p <####>:<####> -v **<Local Directory>**:*<Embedded Path>** --net host rhel8/postgresql-12
ExecReload=-/usr/bin/podman stop "**<PostgreSQL Container Name>**"
ExecReload=-/usr/bin/podman rm "**<PostgreSQL Container Name>**"
ExecStop=-/usr/bin/podman stop "**<PostgreSQL Container Name>**"
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
```
8. The systemd daemon will be reloaded to then start the PostgreSQL service as supported by the added automatic restart configuration:
```bash
$ sudo systemctl daemon-reload

$ sudo systemctl start **<PostgreSQL Container Name>**
```
9.  We’ll conclude by checking the PostgreSQL databases’ service status to determine that the service is actively running:
```bash
$ sudo systemctl status **<PostgreSQL Container Name>**

**<PostgreSQL Container Name>** - PostgreSQL Podman Container
Loaded: loaded (/etc/systemd/system/**<PostgreSQL Container Name>**.service)
Active: active (running)
 ```