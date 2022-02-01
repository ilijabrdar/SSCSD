public class BaremetalPingPxeResource extends BaremetalPxeResourceBase {
    private static final Logger s_logger = Logger.getLogger(BaremetalPingPxeResource.class);
    private static final String Name = "BaremetalPingPxeResource";
    String _storageServer;
    String _pingDir;
    String _share;
    String _dir;
    String _tftpDir;
    String _cifsUserName;
    String _cifsPassword;

    @Override
    public boolean configure(String name, Map<String, Object> params) throws ConfigurationException {
        super.configure(name, params);
        _storageServer = (String)params.get(BaremetalPxeService.PXE_PARAM_PING_STORAGE_SERVER_IP);
        _pingDir = (String)params.get(BaremetalPxeService.PXE_PARAM_PING_ROOT_DIR);
        _tftpDir = (String)params.get(BaremetalPxeService.PXE_PARAM_TFTP_DIR);
        _cifsUserName = (String)params.get(BaremetalPxeService.PXE_PARAM_PING_STORAGE_SERVER_USERNAME);
        _cifsPassword = (String)params.get(BaremetalPxeService.PXE_PARAM_PING_STORAGE_SERVER_PASSWORD);

        if (_podId == null) {
            throw new ConfigurationException("No Pod specified");
        }

        if (_storageServer == null) {
            throw new ConfigurationException("No stroage server specified");
        }

        if (_tftpDir == null) {
            throw new ConfigurationException("No tftp directory specified");
        }

        if (_pingDir == null) {
            throw new ConfigurationException("No PING directory specified");
        }

        if (_cifsUserName == null || _cifsUserName.equalsIgnoreCase("")) {
            _cifsUserName = "xxx";
        }

        if (_cifsPassword == null || _cifsPassword.equalsIgnoreCase("")) {
            _cifsPassword = "xxx";
        }

        String pingDirs[] = _pingDir.split("/");
        if (pingDirs.length != 2) {
            throw new ConfigurationException("PING dir should have format like myshare/direcotry, eg: windows/64bit");
        }
        _share = pingDirs[0];
        _dir = pingDirs[1];

        com.trilead.ssh2.Connection sshConnection = new com.trilead.ssh2.Connection(_ip, 22);

        s_logger.debug(String.format("Trying to connect to PING PXE server(IP=%1$s, username=%2$s, password=%3$s", _ip, _username, "******"));
        try {
            sshConnection.connect(null, 60000, 60000);
            if (!sshConnection.authenticateWithPassword(_username, _password)) {
                s_logger.debug("SSH Failed to authenticate");
                throw new ConfigurationException(String.format("Cannot connect to PING PXE server(IP=%1$s, username=%2$s, password=%3$s", _ip, _username, "******"));
            }

            String cmd = String.format("[ -f /%1$s/pxelinux.0 ] && [ -f /%2$s/kernel ] && [ -f /%3$s/initrd.gz ] ", _tftpDir, _tftpDir, _tftpDir);
            if (!SSHCmdHelper.sshExecuteCmd(sshConnection, cmd)) {
                throw new ConfigurationException("Miss files in TFTP directory at " + _tftpDir + " check if pxelinux.0, kernel initrd.gz are here");
            }

            SCPClient scp = new SCPClient(sshConnection);
            String prepareScript = "scripts/network/ping/prepare_tftp_bootfile.py";
            String prepareScriptPath = Script.findScript("", prepareScript);
            if (prepareScriptPath == null) {
                throw new ConfigurationException("Can not find prepare_tftp_bootfile.py at " + prepareScriptPath);
            }
            scp.put(prepareScriptPath, "/usr/bin/", "0755");

            String userDataScript = "scripts/network/ping/baremetal_user_data.py";
            String userDataScriptPath = Script.findScript("", userDataScript);
            if (userDataScriptPath == null) {
                throw new ConfigurationException("Can not find baremetal_user_data.py at " + userDataScriptPath);
            }
            scp.put(userDataScriptPath, "/usr/bin/", "0755");

            return true;
        } catch (Exception e) {
            throw new ConfigurationException(e.getMessage());
        } finally {
            if (sshConnection != null) {
                sshConnection.close();
            }
        }
    }

    @Override
    public PingCommand getCurrentStatus(long id) {
        com.trilead.ssh2.Connection sshConnection = SSHCmdHelper.acquireAuthorizedConnection(_ip, _username, _password);
        if (sshConnection == null) {
            return null;
        } else {
            SSHCmdHelper.releaseSshConnection(sshConnection);
            return new PingRoutingCommand(getType(), id, new HashMap<String, HostVmStateReportEntry>());
        }
    }

    protected PreparePxeServerAnswer execute(PreparePxeServerCommand cmd) {
        com.trilead.ssh2.Connection sshConnection = new com.trilead.ssh2.Connection(_ip, 22);
        try {
            sshConnection.connect(null, 60000, 60000);
            if (!sshConnection.authenticateWithPassword(_username, _password)) {
                s_logger.debug("SSH Failed to authenticate");
                throw new ConfigurationException(String.format("Cannot connect to PING PXE server(IP=%1$s, username=%2$s, password=%3$s", _ip, _username, _password));
            }

            String script =
                String.format("python /usr/bin/prepare_tftp_bootfile.py restore %1$s %2$s %3$s %4$s %5$s %6$s %7$s %8$s %9$s %10$s %11$s", _tftpDir, cmd.getMac(),
                    _storageServer, _share, _dir, cmd.getTemplate(), _cifsUserName, _cifsPassword, cmd.getIp(), cmd.getNetMask(), cmd.getGateWay());
            if (!SSHCmdHelper.sshExecuteCmd(sshConnection, script)) {
                return new PreparePxeServerAnswer(cmd, "prepare PING at " + _ip + " failed, command:" + script);
            }
            s_logger.debug("Prepare Ping PXE server successfully");

            return new PreparePxeServerAnswer(cmd);
        } catch (Exception e) {
            s_logger.debug("Prepare PING pxe server failed", e);
            return new PreparePxeServerAnswer(cmd, e.getMessage());
        } finally {
            if (sshConnection != null) {
                sshConnection.close();
            }
        }
    }

    protected Answer execute(PrepareCreateTemplateCommand cmd) {
        com.trilead.ssh2.Connection sshConnection = new com.trilead.ssh2.Connection(_ip, 22);
        try {
            sshConnection.connect(null, 60000, 60000);
            if (!sshConnection.authenticateWithPassword(_username, _password)) {
                s_logger.debug("SSH Failed to authenticate");
                throw new ConfigurationException(String.format("Cannot connect to PING PXE server(IP=%1$s, username=%2$s, password=%3$s", _ip, _username, _password));
            }

            String script =
                String.format("python /usr/bin/prepare_tftp_bootfile.py backup %1$s %2$s %3$s %4$s %5$s %6$s %7$s %8$s %9$s %10$s %11$s", _tftpDir, cmd.getMac(),
                    _storageServer, _share, _dir, cmd.getTemplate(), _cifsUserName, _cifsPassword, cmd.getIp(), cmd.getNetMask(), cmd.getGateWay());
            if (!SSHCmdHelper.sshExecuteCmd(sshConnection, script)) {
                return new Answer(cmd, false, "prepare for creating template failed, command:" + script);
            }
            s_logger.debug("Prepare for creating template successfully");

            return new Answer(cmd, true, "Success");
        } catch (Exception e) {
            s_logger.debug("Prepare for creating baremetal template failed", e);
            return new Answer(cmd, false, e.getMessage());
        } finally {
            if (sshConnection != null) {
                sshConnection.close();
            }
        }
    }

    @Override
    public Answer executeRequest(Command cmd) {
        if (cmd instanceof PreparePxeServerCommand) {
            return execute((PreparePxeServerCommand)cmd);
        } else if (cmd instanceof PrepareCreateTemplateCommand) {
            return execute((PrepareCreateTemplateCommand)cmd);
        } else if (cmd instanceof VmDataCommand) {
            return execute((VmDataCommand)cmd);
        } else {
            return super.executeRequest(cmd);
        }
    }

    private Answer execute(VmDataCommand cmd) {
        com.trilead.ssh2.Connection sshConnection = new com.trilead.ssh2.Connection(_ip, 22);
        try {
            List<String[]> vmData = cmd.getVmData();
            StringBuilder sb = new StringBuilder();
            for (String[] data : vmData) {
                String folder = data[0];
                String file = data[1];
                String contents = (data[2] == null) ? "none" : data[2];
                sb.append(cmd.getVmIpAddress());
                sb.append(",");
                sb.append(folder);
                sb.append(",");
                sb.append(file);
                sb.append(",");
                sb.append(contents);
                sb.append(";");
            }
            String arg = org.apache.commons.lang.StringUtils.stripEnd(sb.toString(), ";");

            sshConnection.connect(null, 60000, 60000);
            if (!sshConnection.authenticateWithPassword(_username, _password)) {
                s_logger.debug("SSH Failed to authenticate");
                throw new ConfigurationException(String.format("Cannot connect to PING PXE server(IP=%1$s, username=%2$s, password=%3$s", _ip, _username, _password));
            }

            String script = String.format("python /usr/bin/baremetal_user_data.py '%s'", arg);
            if (!SSHCmdHelper.sshExecuteCmd(sshConnection, script)) {
                return new Answer(cmd, false, "Failed to add user data, command:" + script);
            }

            return new Answer(cmd, true, "Success");
        } catch (Exception e) {
            s_logger.debug("Prepare for creating baremetal template failed", e);
            return new Answer(cmd, false, e.getMessage());
        } finally {
            if (sshConnection != null) {
                sshConnection.close();
            }
        }
    }
}